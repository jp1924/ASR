import re
from typing import Callable, Literal, Tuple
from unicodedata import normalize

normal_dual_bracket_regex = re.compile(r"\(([^()]+)\)/\(([^()]+)\)")

number_regex = re.compile(r"[0-9]")
english_regex = re.compile(r"[A-Za-z]")

# unit
percentage_regex = re.compile(r"[0-9 ](퍼센트|프로|퍼)")

milli_meter_regex = re.compile(r"[0-9 ](밀리미터|미리미터|밀리|미리)")
centi_meter_regex = re.compile(r"[0-9 ](센치미터|센티미터|센치|센티)")
meter_regex = re.compile(r"[0-9 ](미터|미타|메타|메다)")
kilo_meter_regex = re.compile(r"[0-9 ](킬로미터|킬로메타|키로메타|키로미타)")

milli_liter_regex = re.compile(r"[0-9 ](밀리리터|미리리터)")
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

noise_filter_regex = re.compile(r"(u/|b/|l/|o/|n/|\*|\+|@웃음|@목청|@박수|@노래|/\(noise\)|/\(bgm\))")

term_extract_regex = re.compile(r"\(@([^\(\)\/]+)\)")


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


def get_transcript_pair(sentence: str) -> Tuple[str, str, str]:
    sentence = normalize("NFC", sentence)
    sentence = sentence.strip()

    spelling = normal_dual_transcript_extractor(sentence, "left", unit_system_normalize)
    phonetic = normal_dual_transcript_extractor(sentence, "right")

    if (spelling == sentence) or (phonetic == spelling):
        return ("", "", "")

    return (spelling, phonetic, sentence)


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


def preprocess_sentence(example):
    data_ls = list()
    for sentence in example:
        if "idiom" in sentence:
            continue
        # TODO: noise filtering을 하지 않고 이중 전사문을 추출하기 때문에 어떤 문제가 발생할 지 예상이 안됨 확인 필요
        spelling, phonetic, sentence = get_transcript_pair(sentence)

        if not spelling:
            continue

        clean_spelling = noise_filter_regex.sub("", spelling)
        clean_spelling = term_extractor(clean_spelling)

        clean_phonetic = noise_filter_regex.sub("", phonetic)
        clean_phonetic = term_extractor(clean_phonetic)
        if not (number_regex.findall(clean_spelling) or english_regex.findall(clean_phonetic)):
            continue

        if number_regex.findall(clean_phonetic) or english_regex.findall(clean_phonetic):
            continue

        check_phonetic_bracket = ("(" in clean_phonetic) or (")" in clean_phonetic)
        check_spelling_bracket = ("(" in clean_spelling) or (")" in clean_spelling)
        if check_phonetic_bracket or check_spelling_bracket:
            continue

        data = {
            "spelling": spelling,
            "phonetic": phonetic,
            "sentence": sentence,
        }

        data_ls.append(data)
    return data_ls
