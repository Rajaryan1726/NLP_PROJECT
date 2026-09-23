import pytest

import verifier


def test_perfect_summary_scores_10():
    s = verifier.compute_scores(["entailed"] * 4, n_facts=5, n_matched=5)
    assert s["factuality_score"] == 10.0


def test_weighted_formula():
    # claim_score 0.5, entity_score 0.75 -> 10 * (0.6*0.5 + 0.4*0.75) = 6.0
    s = verifier.compute_scores(["entailed", "neutral", "entailed", "contradicted"], n_facts=4, n_matched=3)
    assert s["claim_score"] == 0.5
    assert s["entity_score"] == 0.75
    assert s["factuality_score"] == pytest.approx(6.0)


def test_no_facts_means_nothing_mismatched():
    s = verifier.compute_scores(["entailed", "unsupported"], n_facts=0, n_matched=0)
    assert s["entity_score"] == 1.0
    assert s["factuality_score"] == pytest.approx(10 * (0.6 * 0.5 + 0.4))


def test_no_claims_scores_zero_claim_part():
    s = verifier.compute_scores([], n_facts=0, n_matched=0)
    assert s["claim_score"] == 0.0


def test_decide_prefers_entailment_over_contradiction():
    premises = [("a", {"chunk_id": 0}), ("b", {"chunk_id": 1}), ("a b", {"chunk_id": 0})]
    probs = [{"entailment": 0.1, "neutral": 0.1, "contradiction": 0.8},
             {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},
             {"entailment": 0.4, "neutral": 0.5, "contradiction": 0.1}]
    verdict, conf, chunk = verifier._decide(premises, probs)
    assert verdict == "entailed" and conf == 0.9 and chunk["chunk_id"] == 1


def test_decide_neutral_when_nothing_decisive():
    premises = [("a", {"chunk_id": 0})]
    verdict, _, _ = verifier._decide(premises, [{"entailment": 0.2, "neutral": 0.7, "contradiction": 0.1}])
    assert verdict == "neutral"


def test_premises_include_single_sentences():
    chunk = {"chunk_id": 0, "text_en": "Messages were posted. Hackers changed the bio. They added a photo."}
    other = {"chunk_id": 1, "text_en": "Police are investigating."}
    texts = [p for p, _ in verifier._premises([chunk, other])]
    assert "Messages were posted." in texts        # sentence level
    assert chunk["text_en"] in texts               # whole chunk
    assert chunk["text_en"] + " " + other["text_en"] in texts  # top two joined


def test_entity_in_source_uses_translation_and_fuzzy_spelling():
    source = "केंद्र सरकार ने 'विद्या सेतु' योजना शुरू की। शिक्षा मंत्री धर्मेंद्र प्रधान ने बताया।"
    source_en = "The central government launched the 'Vidya Setu' scheme. Education Minister Dharmendra Pradhan said."
    assert verifier.entity_in_source("Vidya Setu", [source, source_en])
    assert verifier.entity_in_source("Dharmendra Pradhaan", [source, source_en])  # spelling variant
    assert not verifier.entity_in_source("Narendra Modi", [source, source_en])
    assert not verifier.entity_in_source("Mumbai", [source, source_en])
