from icare_poc.app import _finished_thinking, _thinking_html
from icare_poc.theme import ACCENT, DEFAULT_THEME, css, normalize_theme


def test_theme_tokens_cover_both_palettes():
    assert normalize_theme(None) == DEFAULT_THEME
    assert normalize_theme("nuit") == DEFAULT_THEME
    for name in ("clair", "sombre"):
        sheet = css(name)
        assert ACCENT in sheet
        assert "--icare-bg:" in sheet
        assert ".icare-source" in sheet
        assert ".icare-think" in sheet
        assert "#MainMenu" in sheet


def test_thinking_flow_shows_current_step_and_keeps_the_trail():
    html = _thinking_html({"think": "done", "analyse": "running"})
    assert "Réflexion" in html
    assert "Analyse" in html
    assert "icare-orb" in html
    assert "Lecture" not in html
    assert _finished_thinking({"think": "done", "analyse": "running", "lecture": "done"}) == [
        {"key": "think", "label": "Réflexion"},
        {"key": "lecture", "label": "Lecture"},
    ]
