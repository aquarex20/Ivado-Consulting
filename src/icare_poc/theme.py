"""Palettes and chrome for the client-facing Streamlit demo."""

from __future__ import annotations

THEMES = ("clair", "sombre")
DEFAULT_THEME = "clair"
ACCENT = "#3A6B7C"

_PALETTES: dict[str, dict[str, str]] = {
    "clair": {
        "bg": "#F4F1EB",
        "surface": "#FFFCF7",
        "surface2": "#EFEAE2",
        "sidebar": "#EBE6DE",
        "text": "#1C1916",
        "muted": "#6A645C",
        "border": "#D6CFC4",
        "accent": ACCENT,
        "accent_soft": "#DCE8EC",
        "user": "#E7E1D6",
        "error": "#8B3A2F",
        "shadow": "0 10px 30px rgba(28, 25, 22, 0.06)",
    },
    "sombre": {
        "bg": "#12151A",
        "surface": "#1B2027",
        "surface2": "#232A33",
        "sidebar": "#161A20",
        "text": "#ECE8E2",
        "muted": "#9A948C",
        "border": "#3C4652",
        "accent": ACCENT,
        "accent_soft": "#1E2C32",
        "user": "#232A33",
        "error": "#D9897C",
        "shadow": "0 12px 32px rgba(0, 0, 0, 0.28)",
    },
}


def normalize_theme(value: str | None) -> str:
    return value if value in _PALETTES else DEFAULT_THEME


def css(theme: str) -> str:
    palette = _PALETTES[normalize_theme(theme)]
    tokens = "\n".join(f"  --icare-{name.replace('_', '-')}: {value};" for name, value in palette.items())
    return f"""<style>
:root {{
{tokens}
  --icare-sans: "Avenir Next", "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --icare-serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
}}

html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: var(--icare-bg) !important;
  color: var(--icare-text) !important;
  font-family: var(--icare-sans);
}}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"],
#MainMenu, footer, header[data-testid="stHeader"] {{
  display: none !important;
}}
.stAppDeployButton, div[data-testid="stHeader"] {{
  visibility: hidden;
  height: 0;
}}
.block-container {{
  padding-top: 1.4rem;
  padding-bottom: 5rem;
  max-width: 1120px;
}}
[data-testid="stSidebar"] {{
  background: var(--icare-sidebar) !important;
  border-right: 1px solid var(--icare-border);
}}
[data-testid="stSidebar"] * {{
  color: var(--icare-text);
}}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
  color: var(--icare-text) !important;
}}
[data-testid="stSidebar"] .stTextInput input {{
  background: var(--icare-surface);
  color: var(--icare-text);
  border: 1px solid var(--icare-border);
}}
.stApp button,
[data-testid="stSidebar"] .stButton button {{
  background-color: var(--icare-surface) !important;
  color: var(--icare-text) !important;
  border: 1px solid var(--icare-border) !important;
  border-radius: 10px;
  font-weight: 500;
  text-align: left;
  white-space: normal;
  height: auto;
  min-height: 2.6rem;
  line-height: 1.35;
}}
.stApp button p,
.stApp button span {{
  color: inherit !important;
}}
.stApp button:hover {{
  border-color: var(--icare-accent) !important;
  color: var(--icare-accent) !important;
}}
.stApp [data-testid="stMainBlockContainer"] .stButton button,
.stApp .main .stButton button {{
  width: 100%;
  padding: 0.7rem 0.95rem;
}}
.st-key-theme-clair .stButton button,
.st-key-theme-sombre .stButton button {{
  width: 2.45rem;
  min-width: 2.45rem;
  height: 2.45rem;
  min-height: 2.45rem;
  padding: 0;
  border-radius: 999px;
  text-align: center;
  white-space: nowrap;
  font-size: 1.05rem;
  line-height: 1;
}}
.st-key-theme-clair .stButton button[kind="primary"],
.st-key-theme-sombre .stButton button[kind="primary"],
.st-key-theme-clair .stButton button[data-testid="stBaseButton-primary"],
.st-key-theme-sombre .stButton button[data-testid="stBaseButton-primary"] {{
  background-color: var(--icare-accent) !important;
  color: #F7F4EE !important;
  border-color: var(--icare-accent) !important;
}}
.st-key-theme-clair .stButton button[kind="primary"]:hover,
.st-key-theme-sombre .stButton button[kind="primary"]:hover,
.st-key-theme-clair .stButton button[data-testid="stBaseButton-primary"]:hover,
.st-key-theme-sombre .stButton button[data-testid="stBaseButton-primary"]:hover {{
  color: #F7F4EE !important;
}}
[data-testid="stBottom"],
[data-testid="stBottom"] > div {{
  background-color: var(--icare-bg) !important;
  background: var(--icare-bg) !important;
}}
[data-testid="stChatInput"] > div {{
  background-color: var(--icare-surface) !important;
  border: 1px solid var(--icare-border) !important;
}}
[data-testid="stChatInput"],
[data-testid="stChatInput"] textarea,
[data-testid="stSidebar"] input,
.stApp input,
.stApp textarea {{
  background-color: var(--icare-surface) !important;
  color: var(--icare-text) !important;
  border-color: var(--icare-border) !important;
}}
[data-testid="stChatInput"] textarea::placeholder,
.stApp input::placeholder {{
  color: var(--icare-muted) !important;
}}
[data-testid="stExpander"] {{
  background: var(--icare-surface);
  border: 1px solid var(--icare-border);
  border-radius: 12px;
}}
[data-testid="stExpander"] details,
[data-testid="stExpanderDetails"],
[data-testid="stExpander"] [data-testid="stCode"],
[data-testid="stExpander"] pre {{
  overflow: visible !important;
  max-height: none !important;
  height: auto !important;
  white-space: pre-wrap !important;
  background: transparent !important;
}}

.icare-header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1.5rem;
  margin-bottom: 1.75rem;
}}
.icare-brand {{
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}}
.icare-name {{
  font-family: var(--icare-serif);
  font-size: 1.7rem;
  letter-spacing: 0.04em;
  margin: 0;
}}
.icare-tagline {{
  color: var(--icare-muted);
  font-size: 1.02rem;
  margin: 0;
}}
.icare-header-tools {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
  justify-content: flex-end;
}}
.icare-badge {{
  display: inline-block;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  background: var(--icare-accent-soft);
  color: var(--icare-accent);
  font-size: 0.78rem;
  letter-spacing: 0.02em;
}}
.icare-theme-hint {{
  color: var(--icare-muted);
  font-size: 0.78rem;
}}
.icare-starter-label {{
  color: var(--icare-muted);
  font-size: 0.86rem;
  margin: 0 0 0.7rem;
}}
.icare-answer {{
  font-family: var(--icare-serif);
  font-size: 1.35rem;
  line-height: 1.45;
  margin: 0.15rem 0 0.9rem;
}}
.icare-user {{
  background: var(--icare-user);
  border-radius: 14px;
  padding: 0.7rem 0.95rem;
  margin: 0 0 1.1rem;
  color: var(--icare-text);
}}
.icare-error {{
  color: var(--icare-error);
  font-family: var(--icare-serif);
  font-size: 1.15rem;
  margin: 0.15rem 0 0.9rem;
}}
.icare-source {{
  background: var(--icare-surface);
  border: 1px solid var(--icare-border);
  border-left: 3px solid var(--icare-accent);
  border-radius: 12px;
  padding: 0.95rem 1.1rem 1.05rem;
  box-shadow: var(--icare-shadow);
  margin: 0 0 1.4rem;
}}
.icare-source-kicker {{
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--icare-accent);
  margin: 0 0 0.35rem;
}}
.icare-source-meta {{
  color: var(--icare-muted);
  font-size: 0.86rem;
  margin: 0 0 0.7rem;
}}
.icare-reference {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  overflow: visible;
  max-height: none;
  margin: 0;
  color: var(--icare-text);
  line-height: 1.5;
}}
.icare-source-empty {{
  color: var(--icare-muted);
  font-size: 0.9rem;
  margin: 0;
}}
.icare-think {{
  margin: 0.15rem 0 1.15rem;
}}
.icare-think-now {{
  display: flex;
  align-items: center;
  gap: 0.7rem;
  min-height: 1.6rem;
  margin-bottom: 0.55rem;
}}
.icare-orb {{
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: 1.5px solid var(--icare-border);
  border-top-color: var(--icare-accent);
  animation: icare-spin 0.85s linear infinite;
  flex-shrink: 0;
}}
.icare-think-label {{
  font-size: 0.95rem;
  color: var(--icare-muted);
  animation: icare-breathe 1.35s ease-in-out infinite;
}}
.icare-think-trail {{
  list-style: none;
  margin: 0;
  padding: 0;
}}
.icare-think-trail li {{
  display: flex;
  align-items: center;
  gap: 0.55rem;
  color: var(--icare-muted);
  font-size: 0.84rem;
  padding: 0.18rem 0;
}}
.icare-think-trail li.is-run {{
  color: var(--icare-text);
}}
.icare-think-trail li.is-err {{
  color: var(--icare-error);
}}
.icare-think-ico {{
  width: 16px;
  height: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}}
.icare-think-ico svg {{
  width: 14px;
  height: 14px;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}}
@keyframes icare-spin {{
  to {{ transform: rotate(360deg); }}
}}
@keyframes icare-breathe {{
  0%, 100% {{ opacity: 0.42; }}
  50% {{ opacity: 1; }}
}}
.icare-guarantees {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.7rem;
  margin: 1.6rem 0 1rem;
}}
.icare-guarantee {{
  background: var(--icare-surface);
  border: 1px solid var(--icare-border);
  border-radius: 12px;
  padding: 0.85rem 0.95rem;
}}
.icare-guarantee strong {{
  display: block;
  font-size: 0.84rem;
  margin-bottom: 0.3rem;
}}
.icare-guarantee p {{
  margin: 0;
  color: var(--icare-muted);
  font-size: 0.8rem;
  line-height: 1.4;
}}
.icare-roadmap {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.7rem;
}}
.icare-road-card {{
  background: var(--icare-surface2);
  border: 1px dashed var(--icare-border);
  border-radius: 12px;
  padding: 0.85rem 0.95rem;
  pointer-events: none;
}}
.icare-road-card strong {{
  display: block;
  font-size: 0.84rem;
  margin-bottom: 0.3rem;
}}
.icare-road-card p {{
  margin: 0;
  color: var(--icare-muted);
  font-size: 0.8rem;
  line-height: 1.4;
}}
.icare-rail {{
  width: 3px;
  min-height: 2.4rem;
  background: var(--icare-accent);
  border-radius: 2px;
  margin-top: 0.35rem;
}}
.icare-rail-off {{
  width: 3px;
  min-height: 2.4rem;
  margin-top: 0.35rem;
}}
.icare-side-title {{
  font-family: var(--icare-serif);
  font-size: 1.15rem;
  margin: 0 0 0.8rem;
}}
.icare-side-next {{
  margin-top: 1.4rem;
}}
.icare-section {{
  margin: 0 0 0.7rem;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--icare-muted);
}}
.icare-case {{
  background: var(--icare-surface);
  border: 1px solid var(--icare-border);
  border-radius: 14px;
  padding: 1.15rem 1.25rem 0.4rem;
  box-shadow: var(--icare-shadow);
  margin-bottom: 1.3rem;
}}
.icare-case-top {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}}
.icare-case-id {{
  margin: 0;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.06em;
  color: var(--icare-muted);
  font-size: 0.82rem;
}}
.icare-status {{
  border: 1px solid var(--icare-border);
  border-radius: 999px;
  padding: 0.15rem 0.6rem;
  font-size: 0.75rem;
  color: var(--icare-accent);
  background: var(--icare-accent-soft);
}}
.icare-case-name {{
  font-family: var(--icare-serif);
  font-size: 1.7rem;
  font-weight: 500;
  margin: 0.25rem 0 0.9rem;
}}
.icare-fields {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.7rem 1.4rem;
  margin-bottom: 0.9rem;
}}
.icare-field span {{
  display: block;
  color: var(--icare-muted);
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.icare-field strong {{
  font-weight: 500;
}}
.icare-pieces-label {{
  margin: 0 0 0.3rem;
  color: var(--icare-muted);
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.icare-pieces {{
  margin: 0 0 0.8rem;
  padding-left: 1.1rem;
  color: var(--icare-text);
}}
@media (max-width: 800px) {{
  .icare-header {{ flex-direction: column; }}
  .icare-guarantees, .icare-roadmap {{
    grid-template-columns: 1fr;
  }}
}}
</style>
"""
