import math
import re
from typing import Callable, Literal, Optional
from unicodedata import normalize

import librosa
import numpy as np

from transformers import PretrainedConfig


# 해당 모델은 한국어를 소리 그대로 전사하는 것에 목표가 있음
# 전사된 문장에 중국어, 일본어가 들어가 있으면 정상적이지 않은 데이터라 간주하고 필터링 함.

# bracket
deliminator = r"[\/\?\|\<\>\.\:\;\"'\`\!]"  # NOTE: 원래는 "/"가 맞으나 전사 오류로 인해 ? : ! 가 들어가는 경우도 있음.
left_bracket = r".?([^\(\)/]+)\)"  # NOTE ".?"가 들어가 있는 이유는 종종 ()/)나 ()/+)d와 같이 되어 있는 경우가 있음.
right_bracket = r"\(([^\(\)/]+).?"
unnormal_dual_bracket_regex = re.compile(rf"{right_bracket}{deliminator}{left_bracket}")
normal_dual_bracket_regex = re.compile(r"\(([^()]+)\)/\(([^()]+)\)")

# unit
percentage_regex = re.compile(r"[0-9 ](퍼센트|프로|퍼)")

milli_meter_regex = re.compile(r"[0-9 ](밀리미터|미리미터|밀리|미리)")
centi_meter_regex = re.compile(r"[0-9 ](센치미터|센티미터|센치|센티)")
meter_regex = re.compile(r"[0-9 ](미터|미타|메타|메다)")
kilo_meter_regex = re.compile(r"[0-9 ](킬로미터|킬로메타|키로메타|키로미타)")

milli_liter_regex = re.compile(r"[0-9 ](밀리리터|미리리터|밀리리타|미리리타)")
liter_regex = re.compile(r"[0-9 ](리터|리타)")
kilo_liter_regex = re.compile(r"[0-9 ](킬로리터|키로리터|킬로리타|키로리타)")

milli_gram_regex = re.compile(r"[0-9 ](밀리그람|밀리그램|미리그람|미리그램)")
gram_regex = re.compile(r"[0-9 ](그램|그람)")
kilo_gram_regex = re.compile(r"[0-9 ](킬로그램|킬로그람|키로그램|키로그람)")

kilo_byte_regex = re.compile(r"[0-9 ]((킬로 바이트)|(킬로바이트))")
mega_byte_regex = re.compile(r"[0-9 ]((메가 바이트)|(메가바이트))")
giga_byte_regex = re.compile(r"[0-9 ]((기가 바이트)|(기가바이트))")
tera_byte_regex = re.compile(r"[0-9 ]((테라 바이트)|(테라바이트))")
peta_byte_regex = re.compile(r"[0-9 ]((페타 바이트)|(페타바이트))")
exa_byte_regex = re.compile(r"[0-9 ]((엑사 바이트)|(엑사바이트))")
zeta_byte_regex = re.compile(r"[0-9 ]((제타 바이트)|(제타바이트))")
# NOTE: 킬로, 밀리 그람에 대해서는 추가할 지 말지 고민임. 해당 단어와 겹치거나 포함된 단어가 존재할 수 있기 때문에 생각해 봐야 할 듯
#       SI 단위계는 추가는 했으나, US 단위계는 추가하지 않음. 이건 어떻게 해야할지 고민해봐야 할 듯.
#       밀리암페어, 밀리볼트 와 같은 전문적인 단위계는 MM암페어 MM초 이런식으로 번역됨. 이것도 어떻게 처리해야 할지 고민
#       켈빈, 테라, 기가는 패스

# noise & special
double_space_regex = re.compile("([ ]{2,})")
special_char_regex = re.compile(r"""([\@\#\$\^\&\*\~\`\|\-\_\=\+\;\:\'\"\,\<\>\/\{\}\[\]])""")
term_extract_regex = re.compile(r"\(@([^\(\)\/]+)\)")
noise_filter_regex = re.compile(r"(u/|b/|l/|o/|n/|\*|\+|@웃음|@목청|@박수|@노래|/\(noise\)|/\(bgm\))")
bracket_detector = re.compile(r"(\(|\))")

vocab_allow_regex = re.compile(r"[가-힣A-Z0-9\.\% ]")

filtered_language_regex = re.compile(r"[一-龥々〆〤ァ-ヴーぁ-ゔ]")

# NOTE: 주요 영역별 회의 음성인식 데이터는 도메인 단어 혹은 전문단어를 ()/(idiom)과 같이 한다. 여기서 전문 단어를 추출하기 위해서 다음과 같은 정규식을 사용한다.
unidentification_filter_regex = re.compile(
    r"@이름[0-9]+|@상호명[0-9]+|@전화번호[0-9]+|@카드번호[0-9]+|@주민번호[0-9]+|@주소[0-9]+|@정당[0-9]+"
)

space_norm: str = lambda x: double_space_regex.sub(" ", x).strip()
special_char_norm: str = lambda x: special_char_regex.sub("", x)


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


def term_extractor(script: str) -> str:
    bracket_iter = term_extract_regex.finditer(script)
    select_side = 0
    diff = 0
    for idiom in bracket_iter:
        groups = idiom.groups()
        start_idx, end_idx = idiom.span()
        transcript_section = script[start_idx + diff : end_idx + diff]

        script = script[: start_idx + diff] + groups[select_side] + script[end_idx + diff :]
        diff = -(len(transcript_section)) + (len(groups[0]) + diff)

    return script


# TODO: 단위를 맞춰주는게 필요한지는 테스트 필요
def unit_system_normalize(script: str) -> str:
    # 근데 아무리 생각해도 이건 미친짓이긴 한데 그냥 자기 만족용으로 만듬 ㅋㅋㅋ
    percentage_unit = percentage_regex.search(script)
    if percentage_unit:
        start, end = percentage_unit.span(1)  # 0: (전체 범위), 1: (부분 범위)
        script = script[:start] + "%" + script[end:]

    if "암페어" in script:
        return script

    if "볼트" in script:
        return script

    # `바`를 하기엔 겹치는게 너무 많아서 밀리바만 넣었음.
    if "밀리바" in script:
        return script

    if "파스칼" in script:
        return script

    kilo_byte_unit = kilo_byte_regex.search(script)
    if kilo_byte_unit:
        start, end = kilo_byte_unit.span(1)
        script = script[:start] + "KB" + script[end:]

    mega_byte_unit = mega_byte_regex.search(script)
    if mega_byte_unit:
        start, end = mega_byte_unit.span(1)
        script = script[:start] + "MB" + script[end:]

    giga_byte_unit = giga_byte_regex.search(script)
    if giga_byte_unit:
        start, end = giga_byte_unit.span(1)
        script = script[:start] + "GB" + script[end:]

    tera_byte_unit = tera_byte_regex.search(script)
    if tera_byte_unit:
        start, end = tera_byte_unit.span(1)
        script = script[:start] + "TB" + script[end:]

    peta_byte_unit = peta_byte_regex.search(script)
    if peta_byte_unit:
        start, end = peta_byte_unit.span(1)
        script = script[:start] + "PB" + script[end:]

    exa_byte_unit = exa_byte_regex.search(script)
    if exa_byte_unit:
        start, end = exa_byte_unit.span(1)
        script = script[:start] + "EB" + script[end:]

    zeta_byte_unit = zeta_byte_regex.search(script)
    if zeta_byte_unit:
        start, end = zeta_byte_unit.span(1)
        script = script[:start] + "ZB" + script[end:]

    milli_liter_unit = milli_liter_regex.search(script)
    if milli_liter_unit:
        start, end = milli_liter_unit.span(1)
        script = script[:start] + "ML" + script[end:]

    kilo_liter_unit = kilo_liter_regex.search(script)
    if kilo_liter_unit:
        start, end = kilo_liter_unit.span(1)
        script = script[:start] + "KL" + script[end:]

    milli_gram_unit = milli_gram_regex.search(script)
    if milli_gram_unit:
        start, end = milli_gram_unit.span(1)
        script = script[:start] + "MG" + script[end:]

    kilo_gram_unit = kilo_gram_regex.search(script)
    if kilo_gram_unit:
        start, end = kilo_gram_unit.span(1)
        script = script[:start] + "KG" + script[end:]

    kilo_meter_unit = kilo_meter_regex.search(script)
    if kilo_meter_unit:
        start, end = kilo_meter_unit.span(1)
        script = script[:start] + "KM" + script[end:]

    milli_meter_unit = milli_meter_regex.search(script)
    if milli_meter_unit:
        start, end = milli_meter_unit.span(1)
        script = script[:start] + "MM" + script[end:]

    centi_meter_unit = centi_meter_regex.search(script)
    if centi_meter_unit:
        start, end = centi_meter_unit.span(1)
        script = script[:start] + "CM" + script[end:]

    liter_unit = liter_regex.search(script)
    if liter_unit:
        start, end = liter_unit.span(1)
        script = script[:start] + "L" + script[end:]

    gram_unit = gram_regex.search(script)
    if gram_unit:
        start, end = gram_unit.span(1)
        script = script[:start] + "G" + script[end:]

    meter_unit = meter_regex.search(script)
    if meter_unit:
        start, end = meter_unit.span(1)
        script = script[:start] + "M" + script[end:]

    return script


def noise_mark_delete(script: str) -> str:
    return noise_filter_regex.sub("", script)


def unidentification_delete(script: str) -> str:
    return unidentification_filter_regex.sub("", script)


def librosa_silence_filter(audio: np.ndarray, filter_decibel: int = 30) -> np.ndarray:
    idx_list = librosa.effects.split(audio, top_db=filter_decibel)
    split_audio = [audio[start:end] for start, end in idx_list]
    filtered_audio = np.concatenate(split_audio)

    return filtered_audio


def default_sentence_norm(sentence: str) -> str:
    # KsponSpeech 기준
    sentence = normalize("NFC", sentence)
    if "idiom" in sentence:
        # NOTE: idiom 어노테이션 개같이 되어 있어서 그냥 전부 필터링 함.
        # 할꺼면 일관되게 하던지 (.)/(idiom)하고 (idiom)/(.)이게 뭐냐, 짜피 전채 문장에서 54583 정도 밖에 안되서 필터링 하기로 함.
        # default_sentence_norm는 필터링 할 거 다 하고 난 뒤에 괄호가 남아 있으면 전사가 잘못된 것으로 판단해서 전부 필터링 함.
        # 이 코드에선 사용하기가 어려움.
        return ""

    sentence = noise_mark_delete(sentence)
    sentence = unidentification_delete(sentence)

    sentence = sentence.upper()

    sentence = normal_dual_transcript_extractor(sentence, "left", unit_system_normalize)
    sentence = unnormal_dual_transcript_extractor(sentence, "left", unit_system_normalize)

    sentence = term_extractor(sentence)

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
    sentence = special_char_regex.sub(" ", sentence)

    # NOTE: Vocab에 허용되는 문자 이외의 뭔가가 남았다면, 이상한 데이터로 간주하고 필터링 함.
    if vocab_allow_regex.sub("", sentence):
        return ""

    sentence = double_space_regex.sub(" ", sentence)
    sentence = sentence.strip()

    sentence = normalize("NFD", sentence)

    return sentence


def get_feat_extract_output_lengths(
    input_lengths: int,
    config: PretrainedConfig,
    add_adapter: Optional[bool] = None,
) -> int:
    """
    Computes the output length of the convolutional layers
    """

    add_adapter = config.add_adapter if add_adapter is None else add_adapter

    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

        return math.floor((input_length - kernel_size) / stride) + 1

    for kernel_size, stride in zip(config.conv_kernel, config.conv_stride):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

    if add_adapter:
        for _ in range(config.num_adapter_layers):
            input_lengths = _conv_out_length(input_lengths, 1, config.adapter_stride)

    return input_lengths
