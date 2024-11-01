from collections import defaultdict
from typing import List

import numpy as np
from datasets import Dataset
from tqdm import tqdm


def get_packing_strategies(length_ls: List[int], max_seq_len: int, max_seq_per_pack: int) -> defaultdict:
    def add_pack(
        pack: List[int],
        count: int,
        tmp: defaultdict,
        final: defaultdict,
        limit: int,
        offset: int,
    ) -> None:
        if len(pack) == limit or offset == 0:
            final[offset].append((count, pack))
        else:
            tmp[offset].append((count, pack))

    seq_lens, counts = np.unique(length_ls, return_counts=True)
    histogram = np.zeros(max_seq_len, dtype=np.int64)
    histogram[seq_lens - 1] = counts

    reversed_histogram = np.flip(histogram)

    tmp_strategies_per_length = defaultdict(list)
    strategies_per_length = defaultdict(list)

    for i in range(max_seq_len):
        n_sequences_to_bin = reversed_histogram[i]
        length_to_bin = max_seq_len - i
        offset = i + 1  # largest possible offset
        while n_sequences_to_bin > 0:
            if (length_to_bin + offset) in tmp_strategies_per_length:
                # extract shortest pack that will get modified
                n_sequences_to_pack, pack = tmp_strategies_per_length[length_to_bin + offset].pop()
                new_pack = pack + [length_to_bin]
                count = min(n_sequences_to_pack, n_sequences_to_bin)
                if n_sequences_to_pack > n_sequences_to_bin:
                    # old pack gets reduced
                    n_sequences_to_pack -= n_sequences_to_bin
                    tmp_strategies_per_length[length_to_bin + offset].append((n_sequences_to_pack, pack))
                    n_sequences_to_bin = 0
                else:
                    n_sequences_to_bin -= n_sequences_to_pack
                add_pack(new_pack, count, tmp_strategies_per_length, strategies_per_length, max_seq_per_pack, offset)
                # clean up to speed up main key search
                if not tmp_strategies_per_length[length_to_bin + offset]:
                    tmp_strategies_per_length.pop(length_to_bin + offset)
            else:
                offset -= 1
            # Does not fit anywhere. Create new pack.
            if offset < 0:
                add_pack(
                    [length_to_bin],
                    n_sequences_to_bin,
                    tmp_strategies_per_length,
                    strategies_per_length,
                    max_seq_per_pack,
                    i,
                )
                n_sequences_to_bin = 0
    # merge all strategies
    for key in tmp_strategies_per_length:
        strategies_per_length[key].extend(tmp_strategies_per_length[key])

    return strategies_per_length


def get_packing_dataset_idx(length_ls: List[int], strategies_per_length: defaultdict) -> Dataset:
    length_to_indices = {}
    length_array = np.array(length_ls)
    unique_lengths = np.unique(length_array)

    for length in unique_lengths:
        length_to_indices[length] = np.where(length_array == length)[0]

    current_positions = {length: 0 for length in unique_lengths}

    compressed_num = sum(
        strategies_num
        for pad_num, strategies_ls in strategies_per_length.items()
        for strategies_num, seq_ls in strategies_ls
    )

    packing_dataset_ls = []
    with tqdm(total=compressed_num, desc="find_packing_idx") as pbar:
        for pad_num, strategies_ls in strategies_per_length.items():
            for repeat_count, strategies in strategies_ls:
                for _ in range(repeat_count):
                    data_idx_ls = []
                    for length in strategies:
                        indices = length_to_indices[length]
                        current_pos = current_positions[length]

                        while current_pos < len(indices):
                            idx = indices[current_pos]
                            if length_array[idx] != -1:
                                break
                            current_pos += 1

                        if current_pos < len(indices):
                            data_idx = indices[current_pos]
                            length_array[data_idx] = -1
                            data_idx_ls.append(data_idx)
                            current_positions[length] = current_pos + 1

                    packing_dataset_ls.append(
                        {
                            "packing_ls": data_idx_ls,
                            "feat_length_ls": strategies,
                        },
                    )
                    pbar.update(1)

    return Dataset.from_list(packing_dataset_ls)
