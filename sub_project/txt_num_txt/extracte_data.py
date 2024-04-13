from datasets import Dataset, concatenate_datasets, load_dataset
from utils import preprocess_sentence


def main() -> None:
    dataset_name = [
        "jp1924/KoreaSpeech",
        "jp1924/KsponSpeech",
        "jp1924/KconfSpeech",
        "jp1924/KrespSpeech",
        "jp1924/MeetingSpeech",
        "jp1924/BroadcastSpeech",
    ]

    dataset_ls = list()
    for data_name in dataset_name:
        dataset = load_dataset(data_name)
        dataset_ls.extend([dataset[key].select_columns(["sentence"]) for key in dataset])

    dataset_ls = [x.select_columns(["sentence"]) for x in dataset_ls]
    datasets = concatenate_datasets(dataset_ls)

    sentence = [x for x in datasets["sentence"] if ")/(" in x]

    sentence = preprocess_sentence(sentence)

    dataset = Dataset.from_list(sentence)

    # TODO: 나중에 arg로 할 수 있게 만들 것
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(0.1, shuffle=False)

    dataset.push_to_hub("jp1924/TNT_inst")


if "__main__" in __name__:
    main()
