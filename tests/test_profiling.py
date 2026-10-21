from __future__ import annotations

import json

from icare_rules import pipeline as pl
from icare_rules import profiling
from icare_rules.schemas import json_schema

G2800 = (
    "PROGRAMME COMPORTANT DEUX CENT QUARANTE (240) HEURES APRES DES ETUDES DE NIVEAUX VARIES\n"
    "Lorsque le programme complet a été suivi et que le diplôme est obtenu, on accorde seize "
    "trentièmes (16/30) de scolarité qui s’ajoute à la scolarité déjà acquise.\n"
    "Scolarité maximum pour le diplôme = 16/30 d’année de scolarité\n"
    "N.B.(1) Exiger le relevé de notes officiel.\n"
    "N.B.(2) Si le candidat a été admis à ce programme avec des études de même spécialisation, "
    "son cas constitue l'objet d'une règle particulière (voir P0048)."
)


def test_extract_quantities_and_granted():
    labels = {(q.unit, q.value) for q in profiling.extract_quantities(G2800)}
    assert ("heure", 240.0) in labels
    assert ("fraction d'année", round(16 / 30, 4)) in labels

    granted = profiling.extract_granted(G2800)
    assert granted and all(q.unit == "fraction d'année" for q in granted)
    assert all(q.unit != "heure" for q in granted)


def test_granted_ignores_prerequisite_years():
    text = (
        '"Bachelor of Arts", obtenu à la suite de la réussite de deux (2) premières années '
        "du programme; on accorde trois (3) années de scolarité qui s'ajoutent à la scolarité."
    )
    all_q = [q.value for q in profiling.extract_quantities(text)]
    granted = [q.value for q in profiling.extract_granted(text)]
    assert all_q == [2.0, 3.0]
    assert granted == [3.0]


def test_references_documents_and_flags():
    assert profiling.extract_references(G2800 + " Voir G2800 et G-123.", "G2800") == [
        "G0123",
        "P0048",
    ]
    assert profiling.extract_required_documents(G2800) == ["relevé de notes officiel"]


def test_template_signature_groups_variants():
    a = (
        '"Bachelor of Arts (Hebrew University)", précédé de la Licence ès lettres: Maroc; '
        "on accorde deux (2) années de scolarité."
    )
    b = (
        '"Bachelor of Science (McGill)", précédé de la Maîtrise ès lettres: France; '
        "on accorde trois (3) années de scolarité."
    )
    c = "Le candidat doit totaliser au moins 60 crédits pour obtenir une année."
    assert profiling.template_signature(a) == profiling.template_signature(b)
    assert profiling.template_signature(a) != profiling.template_signature(c)


def test_tokenize_strips_elisions_and_stopwords():
    assert profiling.tokenize("L’objet d'équivalence de l'établissement") == [
        "objet",
        "équivalence",
        "établissement",
    ]


def test_tfidf_skips_boilerplate_and_hapax():
    docs = [
        profiling.tokenize(t)
        for t in [
            "diplôme obtenu au Maroc théologie",
            "diplôme obtenu au Maroc",
            "diplôme obtenu en France",
            "diplôme obtenu en France",
            "diplôme obtenu au Liban",
        ]
    ]
    keywords = profiling.tfidf_keywords(docs, top_k=3, min_df=2, max_df_ratio=0.5)
    assert keywords[0] == ["maroc"]  # « théologie » : 1 seul document ; « diplôme » : partout
    assert keywords[2] == ["france"]
    assert keywords[4] == []


def test_profile_pipeline_writes_artefacts(settings):
    pl.run_inventory(settings)
    segments, _ = pl.run_extract_and_segment(settings)
    profiles = profiling.build_profiles(segments)
    assert {p.rule_id for p in profiles} == {"G0001", "P0100#001", "P0100#002", "P0100#003"}
    p1 = next(p for p in profiles if p.rule_id == "P0100#001")
    assert p1.subrule_count == 3
    assert [q["value"] for q in p1.granted] == [11.0]

    paths = profiling.write_profiles(settings, profiles)
    for path in paths.values():
        assert path.exists()
    summary = json.loads(paths["corpus_profile"].read_text(encoding="utf-8"))
    assert summary["totals"]["files"] == 2
    assert summary["totals"]["rules_profiled"] == 4


def test_json_schema_theme_enum():
    assert "enum" not in json_schema()["schema"]["properties"]["theme"]
    themes = ["équivalence de diplôme étranger", "programme court"]
    assert json_schema(themes)["schema"]["properties"]["theme"]["enum"] == themes
