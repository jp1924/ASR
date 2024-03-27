import re
from typing import Callable, Literal
from unicodedata import normalize

import librosa
import numpy as np

# bracket
deliminator = r"[\/\?\|\<\>\.\:\;\"'\`\!]"  # NOTE: 원래는 "/"가 맞으나 전사 오류로 인해 ? : ! 가 들어가는 경우도 있음.
left_bracket = r".?([^\(\)/]+)\)"  # NOTE ".?"가 들어가 있는 이유는 종종 ()/)나 ()/+)d와 같이 되어 있는 경우가 있음.
right_bracket = r"\(([^\(\)/]+).?"
unnormal_dual_bracket_regex = re.compile(rf"{right_bracket}{deliminator}{left_bracket}")
normal_dual_bracket_regex = re.compile(r"\(([^()]+)\)/\(([^()]+)\)")

# unit
percentage_regex = re.compile(r"(프로|퍼센트|퍼)")
meter_regex = re.compile(r"(미터)")
centi_meter_regex = re.compile(r"(센치|센티|센치미터|센티미터)")
kilo_meter_regex = re.compile(r"(킬로미터)")
# NOTE: 킬로, 밀리 그람에 대해서는 추가할 지 말지 고민임. 해당 단어와 겹치거나 포함된 단어가 존재할 수 있기 때문에 생각해 봐야 할 듯

# noise & special
double_space_regex = re.compile("([ ]{2,})")
special_chr_regex = re.compile(r"([\+\*\~\-\#\>\<\;\`\,\@\/\&])")
noise_filter = re.compile(r"(u/|b/|l/|o/|n/|\*|\+)")


space_norm: str = lambda x: double_space_regex.sub(" ", x).strip()
special_char_norm: str = lambda x: special_chr_regex.sub("", x)


def normal_dual_transcript_extractor(
    script: str,
    select_side: Literal["left", "right"] = "left",
    transcript_norm: Callable = None,
) -> str:
    """
    ETRI 전사규칙을 따른다면
        오른쪽: 철사
        왼쪽: 발음

    하지만 ETRI 전사 규칙을 따르지 않는 녀석들도 있어서 사용자가 정하도록 할 수 있도록 함.
    transcript_norm: Callable
    """

    # 비 정상적인 이중 전사 브라켓을 추출 함.
    bracket_iter = normal_dual_bracket_regex.finditer(script)
    select_side = 0 if select_side == "left" else 1

    diff = 0
    for bracket in bracket_iter:
        groups = bracket.groups()
        start_idx, end_idx = bracket.span()

        transcript_section = script[start_idx + diff : end_idx + diff]

        if not normal_dual_bracket_regex.search(transcript_section):
            raise ValueError(
                "이중 전사 구문을 추출하는 과정에서 값이 이상하게 바뀌었습니다."
                f"sentence: {transcript_section}"
            )

        extract_groups = (
            transcript_norm(groups[select_side]) if transcript_norm else groups[select_side]
        )

        script = script[: start_idx + diff] + extract_groups + script[end_idx + diff :]
        diff = -(len(transcript_section)) + (len(extract_groups) + diff)

    return script


def unnormal_dual_transcript_extractor(
    script: str,
    select_side: Literal["left", "right"] = "left",
    transcript_norm: Callable = None,
) -> str:
    """
    ETRI 전사규칙을 따른다면
        오른쪽: 철사
        왼쪽: 발음

    하지만 ETRI 전사 규칙을 따르지 않는 녀석들도 있어서 사용자가 정하도록 할 수 있도록 함.
    transcript_norm: Callable
    """

    # 비 정상적인 이중 전사 브라켓을 추출 함.
    bracket_iter = unnormal_dual_bracket_regex.finditer(script)
    select_side = 0 if select_side == "left" else 1

    diff = 0
    for bracket in bracket_iter:
        groups = bracket.groups()
        start_idx, end_idx = bracket.span()

        transcript_section = script[start_idx + diff : end_idx + diff]

        if not unnormal_dual_bracket_regex.search(transcript_section):
            raise ValueError(
                "이중 전사 구문을 추출하는 과정에서 값이 이상하게 바뀌었습니다."
                f"sentence: {transcript_section}"
            )

        extract_groups = (
            transcript_norm(groups[select_side]) if transcript_norm else groups[select_side]
        )

        script = script[: start_idx + diff] + extract_groups + script[end_idx + diff :]
        diff = -(len(transcript_section)) + (len(extract_groups) + diff)

    return script


# TODO: 단위를 맞춰주는게 필요한지는 테스트 필요
def unit_system_normalize(script: str) -> str:
    script = percentage_regex.sub("%", script)
    script = kilo_meter_regex.sub("KM", script)
    script = centi_meter_regex.sub("CM", script)
    script = meter_regex.sub("M", script)
    return script


def noise_mark_delete(script: str) -> str:
    return noise_filter.sub("", script)


def librosa_silence_filter(audio: np.ndarray, filter_decibel: int = 30) -> np.ndarray:
    idx_list = librosa.effects.split(audio, top_db=filter_decibel)
    split_audio = [audio[start:end] for start, end in idx_list]
    filtered_audio = np.concatenate(split_audio)

    return filtered_audio


def default_sentence_norm(sentence: str) -> str:
    # KsponSpeech 기준
    sentence = noise_mark_delete(sentence)
    sentence = sentence.upper()

    sentence = normal_dual_transcript_extractor(sentence, "left", unit_system_normalize)
    sentence = unnormal_dual_transcript_extractor(sentence, "left", unit_system_normalize)

    sentence = sentence.replace(":", "대")
    sentence = sentence.replace("-", " ")

    sentence = special_char_norm(sentence)
    sentence = space_norm(sentence)

    sentence = normalize("NFD", sentence)

    return sentence
