from findex.tokenize import tokenize


def test_simple_text():
    assert list(tokenize("авіаційна безпека")) == [
        "авіаційна",
        "безпека",
    ]


def test_casefold():
    assert list(tokenize("АВІАЦІЯ Авіація авіація")) == [
        "авіація",
        "авіація",
        "авіація",
    ]


def test_cyrillic():
    assert list(tokenize("Кібербезпека аеропорту")) == [
        "кібербезпека",
        "аеропорту",
    ]


def test_apostrophe():
    assert list(tokenize("комп'ютер об'єкт")) == [
        "комп'ютер",
        "об'єкт",
    ]


def test_hyphen():
    assert list(tokenize("кібер-безпека авіаційно-технічний")) == [
        "кібер",
        "безпека",
        "авіаційно",
        "технічний",
    ]


def test_punctuation():
    assert list(tokenize("Привіт! Як справи? Добре, дякую.")) == [
        "привіт",
        "як",
        "справи",
        "добре",
        "дякую",
    ]


def test_digits():
    assert list(tokenize("Аеропорт 2026 має 15 терміналів")) == [
        "аеропорт",
        "має",
        "терміналів",
    ]


def test_empty_text():
    assert list(tokenize("")) == []

def test_unicode_normalization():
    text = "cafe\u0301"
    assert list(tokenize(text)) == ["café"]