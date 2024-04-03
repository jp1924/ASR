import re
from copy import deepcopy
from typing import Callable, Literal
from unicodedata import normalize

import librosa
import numpy as np

# 해당 모델은 한국어를 소리 그대로 전사하는 것에 목표가 있음
# 전사된 문장에 중국어, 일본어가 들어가 있으면 정상적이지 않은 데이터라 간주하고 필터링 함.

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
special_chr_regex = re.compile(r"""([\@\#\$\^\&\*\~\`\|\-\_\=\+\;\:\'\"\,\<\>\/\{\}\[\]])""")
noise_filter = re.compile(r"(u/|b/|l/|o/|n/|\*|\+)")
bracket_detector = re.compile(r"(\(|\))")

vocab_allow_regex = re.compile(r"[가-힣A-Z0-9\.\% ]")

filtered_language_regex = re.compile(r"[一-龥々〆〤ァ-ヴーぁ-ゔ]")

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
                "이중 전사 구문을 추출하는 과정에서 값이 이상하게 바뀌었습니다." f"sentence: {transcript_section}"
            )

        extract_groups = transcript_norm(groups[select_side]) if transcript_norm else groups[select_side]

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
                "이중 전사 구문을 추출하는 과정에서 값이 이상하게 바뀌었습니다." f"sentence: {transcript_section}"
            )

        extract_groups = transcript_norm(groups[select_side]) if transcript_norm else groups[select_side]

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
    sentence = normalize("NFC", sentence)
    sentence = noise_mark_delete(sentence)
    sentence = sentence.upper()

    sentence = normal_dual_transcript_extractor(sentence, "left", unit_system_normalize)
    sentence = unnormal_dual_transcript_extractor(sentence, "left", unit_system_normalize)

    if bracket_detector.findall(sentence):
        return ""

    if filtered_language_regex.findall(sentence):
        return ""

    # NOTE: 느낌표나 물음표의 대부분은 문장이 끝났을 때 사용하게 됨. 그렇기 때문에 느낌표와 물음표는 마침표로 변환 함.
    sentence = sentence.replace("?", ".")
    sentence = sentence.replace("!", ".")
    sentence = sentence.replace(":", "대")

    # 차라리 띄어쓰는게 더 나을 듯. 특수문자 옆에 띄어쓰기가 깉이 있는 경우 `{ ` -> `  `가 되어서 norm 될 수 있을 듯
    # 다만 이렇지 않은 경우를 함 봐야 알 듯
    sentence = special_chr_regex.sub(" ", sentence)

    # NOTE: Vocab에 허용되는 문자 이외의 뭔가가 남았다면, 이상한 데이터로 간주하고 필터링 함.
    if vocab_allow_regex.sub("", sentence):
        return ""

    sentence = double_space_regex.sub(" ", sentence)
    sentence = sentence.strip()

    sentence = normalize("NFD", sentence)

    return sentence
