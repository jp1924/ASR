import json

from datasets import concatenate_datasets, load_dataset
from utils import default_sentence_norm

from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Processor,
)


def preprocessor(example):
    sentence_ls = example["sentence"]
    sentence_ls = sentence_ls if isinstance(sentence_ls, list) else [sentence_ls]

    normalized_sentence_ls = list()
    for sentence in sentence_ls:
        sentence = default_sentence_norm(sentence)

        if not sentence:
            continue

        normalized_sentence_ls.append(sentence)

    return {"sentence": normalized_sentence_ls}


def main() -> None:
    vocab_file_path = "./vocab.json"
    save_dir_path = "/root/model"

    dataset_repo_ls = [
        "jp1924/KsponSpeech",
        "jp1924/KoreanSpeech",
        "jp1924/KconfSpeech",
        "jp1924/KrespSpeech",
        "jp1924/BroadcastSpeech",
        "jp1924/MeetingSpeech",
    ]
    dataset_ls = list()
    for dataset_repo in dataset_repo_ls:
        dataset = load_dataset(dataset_repo, split="train")
        dataset = dataset.select_columns(["sentence"])
        dataset_ls.append(dataset)

    datasets = concatenate_datasets(dataset_ls)
    datasets = datasets.map(preprocessor, batched=True, num_proc=4)

    sentence = datasets["sentence"]
    vocab = sorted(set("".join(sentence)))

    vocab[0] = "|"
    vocab.insert(0, "<s>")
    vocab.insert(0, "</s>")
    vocab.insert(0, "<unk>")
    vocab.insert(0, "<pad>")

    vocab = {char: token_id for token_id, char in enumerate(vocab)}
    vocab = {"ko": vocab}

    with open(vocab_file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab, ensure_ascii=False, indent=4))

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        replace_word_delimiter_char=" ",
        do_lower_case=False,
        target_lang="ko",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=False,
        do_normalize=True,
    )
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
    processor.save_pretrained(save_dir_path)

    config = Wav2Vec2Config()
    model = Wav2Vec2ForPreTraining(config)


if "__main__" in __name__:
    main()
