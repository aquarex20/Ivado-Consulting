from __future__ import annotations

from icare_rules import pipeline as pl
from icare_rules import reporting


def _run_full(settings, fake_backend):
    pl.run_inventory(settings)
    pl.run_extract_and_segment(settings)
    return pl.run_classify(settings, backend=fake_backend, resume=True)


def test_full_pipeline_produces_valid_records(settings, fake_backend):
    records, failures = _run_full(settings, fake_backend)
    assert failures == []
    # G0001 (agrégat fichier) + 3 sous-règles pour P0100
    assert len(records) == 4
    rule_ids = {r.rule_id for r in records}
    assert rule_ids == {"G0001", "P0100#001", "P0100#002", "P0100#003"}

    # Toutes les catégories doivent être valides
    assert all(r.category in {"Déterministe", "Qualitative", "Hybride"} for r in records)


def test_resume_is_idempotent_and_uses_cache(settings, fake_backend):
    _run_full(settings, fake_backend)
    first_call_count = len(fake_backend.calls)

    # Relance : rien à faire, aucun nouvel appel LLM
    records, failures = pl.run_classify(settings, backend=fake_backend, resume=True)
    assert failures == []
    assert records == []  # aucune nouvelle ligne écrite
    assert len(fake_backend.calls) == first_call_count


def test_cache_hits_on_no_resume(settings, fake_backend):
    _run_full(settings, fake_backend)
    calls_before = len(fake_backend.calls)

    # `no_resume` réécrit tout, mais le cache par hash évite les appels réseau.
    records, failures = pl.run_classify(settings, backend=fake_backend, resume=False)
    assert failures == []
    assert len(records) == 4
    assert all(r.cached for r in records)
    assert len(fake_backend.calls) == calls_before  # cache utilisé


def test_identical_texts_share_cache(settings, fake_backend):
    pl.run_inventory(settings)
    segments, _ = pl.run_extract_and_segment(settings)
    twin = next(s for s in segments if s.rule_id == "P0100#001")
    clone = pl.Segment(**{**twin.to_dict(), "rule_id": "P0100#009", "index": 9})
    records, failures = pl.run_classify(
        settings, segments=[*segments, clone], backend=fake_backend, resume=False
    )
    assert failures == []
    assert len(records) == 5
    assert len(fake_backend.calls) == 4


def test_shuffle_is_reproducible_and_complete(settings, fake_backend):
    pl.run_inventory(settings)
    pl.run_extract_and_segment(settings)
    records, _ = pl.run_classify(settings, backend=fake_backend, resume=False, shuffle_seed=3)
    assert {r.rule_id for r in records} == {"G0001", "P0100#001", "P0100#002", "P0100#003"}
    order_a = [c["rule_id"] for c in fake_backend.calls]
    fake_backend.calls.clear()
    settings.classification.prompt_version = "test-bis"
    pl.run_classify(settings, backend=fake_backend, resume=False, shuffle_seed=3)
    assert [c["rule_id"] for c in fake_backend.calls] == order_a


def test_overlong_llm_output_is_clipped():
    from icare_rules.llm_client import _clip_to_limits
    from icare_rules.schemas import Classification

    parsed = {
        "category": "Hybride",
        "justification": "x" * 900,
        "evidence": [f"e{i}" for i in range(9)],
        "confidence": 1.4,
        "manual_review": False,
        "keywords": [f"k{i}" for i in range(15)],
    }
    c = Classification.model_validate(_clip_to_limits(parsed))
    assert len(c.justification) == 600
    assert len(c.evidence) == 6
    assert len(c.keywords) == 10
    assert c.confidence == 1.0


def test_reporting_creates_all_artefacts(settings, fake_backend):
    _run_full(settings, fake_backend)
    paths = reporting.build_all_reports(settings)
    for name in ("detail", "summary_by_file", "review_queue", "metrics", "attribute_vocabulary"):
        assert paths[name].exists(), name

    metrics = paths["metrics"].read_text(encoding="utf-8")
    assert "distribution" in metrics
    assert "Déterministe" in metrics

    vocab = paths["attribute_vocabulary"].read_text(encoding="utf-8").splitlines()
    assert "dossier_variables,diplome_obtenu,4,1,2,1" in vocab
    assert any(line.startswith("keywords,scolarité,4,") for line in vocab)
