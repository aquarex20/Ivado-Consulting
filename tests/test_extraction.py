from __future__ import annotations

import pytest

from icare_rules import extraction
from icare_rules.extraction import ExtractionError, extract_text


def _write(path, data: bytes):
    path.write_bytes(data)
    return path


def test_wordperfect_routed_to_wpd2text_only(settings, tmp_path, monkeypatch):
    settings.extraction.strategies = ["textutil", "wpd2text"]
    calls = []
    monkeypatch.setattr(extraction, "_run_textutil", lambda *a, **k: calls.append("textutil") or "x" * 50)
    monkeypatch.setattr(
        extraction, "_run_wpd2text", lambda *a, **k: calls.append("wpd2text") or "Règle P2991 lisible."
    )
    wp = _write(tmp_path / "P2991.doc", extraction.WORDPERFECT_MAGIC + b"\x00" * 100)
    result = extract_text(wp, settings)
    assert result.strategy == "wpd2text"
    assert calls == ["wpd2text"]

    calls.clear()
    word = _write(tmp_path / "P0001.doc", b"\xd0\xcf\x11\xe0" + b"\x00" * 100)
    assert extract_text(word, settings).strategy == "textutil"
    assert calls == ["textutil"]


def test_wordperfect_without_strategy_fails_explicitly(settings, tmp_path):
    settings.extraction.strategies = ["textutil"]
    wp = _write(tmp_path / "P2991.doc", extraction.WORDPERFECT_MAGIC + b"\x00" * 100)
    with pytest.raises(ExtractionError, match="WordPerfect"):
        extract_text(wp, settings)


def test_garbled_output_is_rejected(settings, tmp_path, monkeypatch):
    settings.extraction.strategies = ["textutil"]
    monkeypatch.setattr(extraction, "_run_textutil", lambda *a, **k: "Règle\x00\x00\x07" * 20)
    word = _write(tmp_path / "G0001.doc", b"\xd0\xcf\x11\xe0" + b"\x00" * 100)
    with pytest.raises(ExtractionError, match="illisible"):
        extract_text(word, settings)


def test_page_markers_are_stripped(monkeypatch, tmp_path):
    class Completed:
        returncode = 0
        stderr = b""
        stdout = (
            "P2991\n                         Page    1 de 1\n"
            "la règle G 0002, émises le 16 décembre 1983.               1 de 3\n"
            "1-\tne pas apparaître dans la liste\n"
        ).encode()

    monkeypatch.setattr(extraction.shutil, "which", lambda _: "/usr/bin/wpd2text")
    monkeypatch.setattr(extraction.subprocess, "run", lambda *a, **k: Completed())
    text = extraction._run_wpd2text(tmp_path / "x.doc", timeout=5)
    assert "Page" not in text
    assert "1 de 3" not in text
    assert "la règle G 0002, émises le 16 décembre 1983." in text
    assert "1-\tne pas apparaître" in text
