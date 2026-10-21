from __future__ import annotations

from icare_rules.segmentation import (
    iter_segments_for_classification,
    segment_rule,
)


def test_file_without_subrules_yields_only_aggregate(settings):
    text = (
        "Le candidat doit avoir cumulé au moins 45 crédits universitaires "
        "pour être considéré éligible."
    )
    segments = segment_rule(
        parent_rule_id="G0001",
        source_file="corpus/G/G0001.docx",
        source_hash="abc",
        text=text,
        settings=settings,
    )
    assert len(segments) == 1
    assert segments[0].kind == "file"
    assert segments[0].rule_id == "G0001"


def test_file_with_subrules_yields_stable_ids(settings):
    text = (
        "1  Certificat X précédé du Cours secondaire : on accorde onze années.\n"
        "2  Certificat Y accompagné du Diplôme Z : on accorde treize années.\n"
        "3- Diplôme W équivalent au domaine ; totaliser les crédits > 60.\n"
    )
    segments = segment_rule(
        parent_rule_id="P0100",
        source_file="corpus/P/P0100.docx",
        source_hash="def",
        text=text,
        settings=settings,
    )
    subs = [s for s in segments if s.kind == "subrule"]
    assert [s.rule_id for s in subs] == ["P0100#001", "P0100#002", "P0100#003"]
    assert [s.marker for s in subs] == ["1", "2", "3"]


def test_iter_prefers_subrules_over_aggregate(settings):
    text = (
        "1  Sous-règle un avec assez de caractères pour être conservée.\n"
        "2  Sous-règle deux avec elle aussi assez de caractères conservés.\n"
    )
    segments = segment_rule(
        parent_rule_id="P0999",
        source_file="corpus/P/P0999.docx",
        source_hash="ghi",
        text=text,
        settings=settings,
    )
    to_classify = list(iter_segments_for_classification(segments))
    assert all(s.kind == "subrule" for s in to_classify)
    assert len(to_classify) == 2
