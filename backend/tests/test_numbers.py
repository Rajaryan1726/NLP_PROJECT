import pytest

import preprocessing as pp


@pytest.mark.parametrize("text, value", [
    ("2 crore", 20_000_000),
    ("2 करोड़", 20_000_000),
    ("4,500 crore", 45_000_000_000),
    ("5 lakh", 500_000),
    ("4,50,000", 450_000),         # Indian comma grouping
    ("3.5 million", 3_500_000),
    ("20 हजार", 20_000),
    ("1.2 bn", 1_200_000_000),
    ("70%", 70),
])
def test_parse_numbers(text, value):
    [(_, parsed)] = pp.parse_numbers(text)
    assert parsed == pytest.approx(value)


def test_devanagari_digits_after_cleaning():
    [(_, value)] = pp.parse_numbers(pp.clean_text("२ करोड़"))
    assert value == 20_000_000


def test_equivalent_amounts_match():
    [(_, crore)] = pp.parse_numbers("2 crore")
    [(_, million)] = pp.parse_numbers("20 million")
    assert pp.numbers_match(crore, million)
    assert not pp.numbers_match(crore, 2_000_000)


def test_decimal_is_not_split():
    assert [v for _, v in pp.parse_numbers("growth 4.5 percent")] == [4.5]


def test_number_words_source_side():
    assert pp.number_words("अगले दो साल में") == {2.0}


@pytest.mark.parametrize("text, expected", [
    ("15 August 2023", (2023, 8, 15)),
    ("15 अगस्त 2023", (2023, 8, 15)),
    ("15/08/2023", (2023, 8, 15)),
    ("2023-08-15", (2023, 8, 15)),
    ("August 15, 2023", (2023, 8, 15)),
    ("अगस्त 2023", (2023, 8, None)),
])
def test_parse_dates(text, expected):
    [(_, parsed)], _ = pp.parse_dates(text)
    assert parsed == expected


def test_dates_are_removed_before_number_extraction():
    _, rest = pp.parse_dates("15 अगस्त 2023 को 2 करोड़")
    assert [v for _, v in pp.parse_numbers(rest)] == [20_000_000]


def test_dates_match_partial():
    assert pp.dates_match((None, 8, 15), (2023, 8, 15))       # "15 August" vs full date
    assert not pp.dates_match((2023, 8, 16), (2023, 8, 15))
    assert not pp.dates_match((2024, None, None), (2023, 8, 15))
