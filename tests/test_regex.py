from masker.detectors.regex_detector import RegexDetector

R = RegexDetector()


def found(text, label):
    return [f.text for f in R.detect(text, [label])]


def test_email_url_ip():
    assert found("mail me: john.smith@acme.com.", "EMAIL") == ["john.smith@acme.com"]
    assert found("see https://acme.com/a?b=1, or www.x.org.", "URL") == ["https://acme.com/a?b=1", "www.x.org"]
    assert found("server 192.168.1.20.", "IP_ADDRESS") == ["192.168.1.20"]
    assert found("version 999.1.1.1", "IP_ADDRESS") == []


def test_phone_international_local_and_arabic_digits():
    assert found("call +1 (212) 555-0100 now", "PHONE") == ["+1 (212) 555-0100"]
    assert found("الهاتف +963 933 123 456 أو 0944123456.", "PHONE") == ["+963 933 123 456", "0944123456"]
    assert found("رقم ٠٩٤٤١٢٣٤٥٦", "PHONE") == ["٠٩٤٤١٢٣٤٥٦"]
    assert found("in 2024-03-12 we met", "PHONE") == []


def test_card_and_iban_are_validated():
    assert found("card 4111 1111 1111 1111", "CREDIT_CARD") == ["4111 1111 1111 1111"]
    assert found("card 4111 1111 1111 1112", "CREDIT_CARD") == []  # bad Luhn
    assert found("IBAN GB82 WEST 1234 5698 7654 32,", "IBAN") == ["GB82 WEST 1234 5698 7654 32"]
    assert found("IBAN SA0380000000608010167519", "IBAN") == ["SA0380000000608010167519"]
    assert found("IBAN GB00 WEST 1234 5698 7654 32", "IBAN") == []  # bad checksum


def test_dates_english_arabic_numeric():
    assert found("On 12 March 2024, we", "DATE") == ["12 March 2024"]
    assert found("March 5th, 2023 and May 2020", "DATE") == ["March 5th, 2023", "May 2020"]
    assert found("في ١٢ آذار ٢٠٢٤ التقى", "DATE") == ["١٢ آذار ٢٠٢٤"]
    assert found("بتاريخ 15/04/2024.", "DATE") == ["15/04/2024"]
    assert found("شهر رمضان المبارك", "DATE") == []  # month name alone is not a date


def test_id_number():
    assert found("passport number 548211930.", "ID_NUMBER") == ["548211930"]
    assert found("year 2024", "ID_NUMBER") == []


def test_context_decides_phone_vs_id():
    fs = R.detect("ورقم الهوية الوطنية 01020304050 والهاتف 0944123456", ["PHONE", "ID_NUMBER"])
    assert [(f.text, f.label) for f in fs] == [("01020304050", "ID_NUMBER"), ("0944123456", "PHONE")]
    fs = R.detect("الهاتف 0944123456 ورقم الهوية 01020304050", ["ID_NUMBER", "PHONE"])
    assert [(f.text, f.label) for f in fs] == [("0944123456", "PHONE"), ("01020304050", "ID_NUMBER")]
