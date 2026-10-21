"""Backend LLM factice pour dry-run (aucun appel réseau).

Applique des heuristiques simples sur le texte pour retourner une catégorie et
une confiance. Utilisé uniquement pour valider la boucle de bout en bout avant
de brancher les credentials Azure.
"""

from __future__ import annotations

import json
import re

from .llm_client import LLMBackend

_NUMERIC_HINTS = re.compile(
    r"\b(cr[ée]dits?|ann[ée]es? de scolarit[ée]|unit[ée]s?|1/\d+|\d{1,3}\s?%|>=?|<=?)\b",
    re.IGNORECASE,
)
_SEMANTIC_HINTS = re.compile(
    r"\b(é?quivalent|libellé|intitulé|domaine|correspondance|d[ée]nomination|jargon)\b",
    re.IGNORECASE,
)
_HYBRID_HINTS = re.compile(
    r"\b(on accorde|totaliser|somme|dépasse|au moins.*ann[ée]es)\b",
    re.IGNORECASE,
)


class DryRunBackend(LLMBackend):
    """Backend qui classe sans appel réseau, à des fins de démonstration."""

    def complete(self, *, system: str, user: str, deployment: str) -> tuple[str, dict[str, int]]:
        payload = json.loads(user)
        text = payload.get("text", "")
        numeric = bool(_NUMERIC_HINTS.search(text))
        semantic = bool(_SEMANTIC_HINTS.search(text))
        hybrid_direct = bool(_HYBRID_HINTS.search(text))

        if hybrid_direct or (numeric and semantic):
            category = "Hybride"
            confidence = 0.6
        elif numeric and not semantic:
            category = "Déterministe"
            confidence = 0.65
        elif semantic and not numeric:
            category = "Qualitative"
            confidence = 0.6
        else:
            category = "Qualitative"
            confidence = 0.45  # peu confiant → sera flaggé pour révision

        response = {
            "category": category,
            "justification": (
                "Classification heuristique (dry-run) sans appel LLM. "
                "Basée sur la présence de motifs numériques et/ou sémantiques."
            ),
            "evidence": [text[:150]],
            "confidence": confidence,
            "manual_review": confidence < 0.5,
        }
        return json.dumps(response, ensure_ascii=False), {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
