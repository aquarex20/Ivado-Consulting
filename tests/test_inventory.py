from __future__ import annotations

from icare_rules.inventory import build_inventory, summarize


def test_inventory_deduplicates_doc_vs_docx(settings):
    entries = build_inventory(settings)
    ids = {e.rule_id for e in entries}
    assert "G0001" in ids
    assert "P0100" in ids

    # Le corpus doit contenir seulement des .docx (config `include_globs`)
    assert all(e.extension == ".docx" for e in entries)


def test_inventory_summary_counts(settings):
    entries = build_inventory(settings)
    summary = summarize(entries)
    assert summary["by_kind"]["G"] == 1
    assert summary["by_kind"]["P"] == 1
    assert summary["duplicates"] == 0
