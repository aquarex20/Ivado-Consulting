"""Modèles Pydantic pour les sorties structurées LLM."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class Category(str, Enum):
    DETERMINISTE = "Déterministe"
    QUALITATIVE = "Qualitative"
    HYBRIDE = "Hybride"


ShortList = Annotated[list[str], Field(max_length=10)]


class Classification(BaseModel):
    """Sortie attendue du LLM pour une règle donnée."""

    model_config = {"extra": "forbid"}

    category: Category
    justification: Annotated[str, Field(min_length=1, max_length=600)]
    evidence: Annotated[list[str], Field(max_length=6)] = []
    confidence: Annotated[float, Field(ge=0, le=1)]
    manual_review: bool = False

    # Attributs descriptifs (prompt v2+) ; valeurs par défaut pour rester
    # compatible avec les réponses v1 et les backends de test.
    theme: str = ""
    keywords: ShortList = []
    diplomas: ShortList = []
    institutions: ShortList = []
    countries: ShortList = []
    prerequisites: ShortList = []
    granted_value: str | None = None
    dossier_variables: ShortList = []


REQUIRED_FIELDS = [
    "category",
    "justification",
    "evidence",
    "confidence",
    "manual_review",
    "theme",
    "keywords",
    "diplomas",
    "institutions",
    "countries",
    "prerequisites",
    "granted_value",
    "dossier_variables",
]


def json_schema(themes: list[str] | None = None) -> dict:
    """Schema JSON compatible avec l'API `response_format=json_schema` Azure OpenAI.

    Si `themes` est fourni, le champ `theme` est contraint à cette liste fermée ;
    sinon il est libre (phase de découverte).
    """
    string_list = {"type": "array", "items": {"type": "string"}}
    theme: dict = {"type": "string"}
    if themes:
        theme["enum"] = list(themes)
    return {
        "name": "IcareRuleClassification",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": REQUIRED_FIELDS,
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [c.value for c in Category],
                },
                "justification": {"type": "string"},
                "evidence": string_list,
                "confidence": {"type": "number"},
                "manual_review": {"type": "boolean"},
                "theme": theme,
                "keywords": string_list,
                "diplomas": string_list,
                "institutions": string_list,
                "countries": string_list,
                "prerequisites": string_list,
                "granted_value": {"type": ["string", "null"]},
                "dossier_variables": string_list,
            },
        },
    }
