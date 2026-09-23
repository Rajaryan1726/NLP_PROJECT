import preprocessing as pp


def test_clean_text_converts_devanagari_digits_and_whitespace():
    assert pp.clean_text("बजट  ४,५००   करोड़\n\n\nहै") == "बजट 4,500 करोड़\nहै"


def test_detect_language():
    assert pp.detect_language("भारत सरकार ने आज नई योजना शुरू की है।") == "hindi"
    assert pp.detect_language("The government launched a new scheme for students today.") == "english"
    assert pp.detect_language("Sarkar ne aaj nayi scheme launch ki hai aur students ko tablets milenge") == "code_mixed"
    assert pp.detect_language("सरकार ने नई digital education scheme launch की है for rural students") == "code_mixed"


def test_split_sentences_keeps_decimals_and_offsets():
    text = "बजट 4.5 करोड़ है। दूसरा वाक्य! Third one."
    sentences = pp.split_sentences(text)
    assert [s for s, _, _ in sentences] == ["बजट 4.5 करोड़ है।", "दूसरा वाक्य!", "Third one."]
    for s, start, end in sentences:
        assert text[start:end] == s


def test_chunks_have_2_to_3_sentences_and_valid_offsets():
    text = " ".join(f"यह वाक्य संख्या {i} है।" for i in range(1, 8))
    chunks = pp.chunk_text(text)
    assert all(1 <= len(pp.split_sentences(c["text"])) <= 3 for c in chunks)
    assert len(pp.split_sentences(chunks[0]["text"])) == 3
    for c in chunks:
        assert text[c["start"]:c["end"]] == c["text"]
