import json

from datasets import concatenate_datasets, load_dataset
from utils import english_regex, get_transcript_pair, number_regex


def main() -> None:
    def preprocessor(example):
        data_ls = list()
        for sentence in example:
            spelling, phonetic, sentence = get_transcript_pair(sentence)

            if not number_regex.findall(spelling):
                continue

            if number_regex.findall(phonetic):
                continue

            if english_regex.findall(phonetic):
                continue

            if ("(" in spelling) or (")" in spelling):
                continue

            if ("(" in phonetic) or (")" in phonetic):
                continue

            if not spelling:
                continue
            data = {
                "spelling": spelling,
                "phonetic": phonetic,
                "sentence": sentence,
            }

            data_ls.append(data)
        return data_ls

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

    sentence = preprocessor(sentence)

    save_path = "/root/tnt_data.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sentence, f, indent=4, ensure_ascii=False)


if "__main__" in __name__:
    main()

"아니 2개가 뭘 필요해 (네)/(니) 한 모금 나 한 모금 그러면 되지."
"어/ 청년 CEO의 (24시)/(이십 사 시). 네. 그 하루를 함께 만나보시죠."
"어/ 청년 CEO의 (24시)/(이십 사 시). 네. 그 하루를 함께 만나보시죠."
