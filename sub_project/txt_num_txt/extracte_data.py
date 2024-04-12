import json

from datasets import concatenate_datasets, load_dataset
from utils import preprocess_sentence


def main() -> None:
    dataset_name = [
        "jp1924/KoreaSpeech",
        "jp1924/KsponSpeech",
        "jp1924/KconfSpeech",
        "jp1924/KrespSpeech",
    ]

    dataset_ls = list()
    for data_name in dataset_name:
        dataset = load_dataset(data_name)
        dataset_ls.extend([dataset[key].select_columns(["sentence"]) for key in dataset])

    dataset_ls = [x.select_columns(["sentence"]) for x in dataset_ls]
    datasets = concatenate_datasets(dataset_ls)

    sentence = [x for x in datasets["sentence"] if ")/(" in x]

    sentence = preprocess_sentence(sentence)

    save_path = "/root/tnt_data.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sentence, f, indent=4, ensure_ascii=False)


if "__main__" in __name__:
    main()
