---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    background: #ffffff;
    color: #1a1a2e;
  }
  section.title {
    background: #1a1a2e;
    color: #ffffff;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.title h1 {
    font-size: 2em;
    margin-bottom: 0.3em;
  }
  section.title p {
    color: #a0b4d6;
    font-size: 1em;
  }
  h2 {
    color: #1a1a2e;
    border-bottom: 3px solid #e63946;
    padding-bottom: 0.2em;
  }
  .highlight {
    background: #f1faee;
    border-left: 4px solid #457b9d;
    padding: 0.5em 1em;
    margin: 0.5em 0;
    border-radius: 0 6px 6px 0;
  }
  .tag-math   { background:#d4edda; color:#155724; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:0.85em; }
  .tag-sem    { background:#cce5ff; color:#004085; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:0.85em; }
  .tag-hybrid { background:#fff3cd; color:#856404; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:0.85em; }
  table { width:100%; border-collapse:collapse; font-size:0.85em; }
  th { background:#1a1a2e; color:#fff; padding:8px; }
  td { padding:8px; border-bottom:1px solid #ddd; }
  tr:nth-child(even) td { background:#f8f9fa; }
  blockquote {
    border-left: 4px solid #e63946;
    background: #fff5f5;
    padding: 0.6em 1em;
    font-style: italic;
    color: #333;
  }
---

<!-- _class: title -->

# Labellisation des règles
## Pour choisir l'architecture

**Étape de cadrage — avant l'évaluation des dossiers**

---

## Contexte & objectif

Nous construisons un système d'IA capable d'émettre une **décision** à partir de :

- **Règles prédéterminées** — le référentiel métier
- **Dossiers de candidature** — documents, parcours, critères

> **L'enjeu :** adopter une architecture performante — réduire la complexité lorsqu'un agent doit retrouver le contexte de règles applicable à une tâche.

---

## Pourquoi labelliser d'abord ?

Confier **toutes** les règles à un seul agent (recherche + interprétation + verdict) crée :

| Problème | Impact |
|---|---|
| Bruit sémantique | Dégradation de la précision |
| Coût computationnel | Latence et coût élevés |
| Risque d'erreur | Mélange calcul & interprétation |

**Solution :** cartographier le référentiel **avant** de figer la structure d'exécution — pour savoir quelle part relève du calcul et quelle part relève du sémantique.

---

## Trois catégories de règles

<br>

| Catégorie | Nature | Traitement |
|---|---|---|
| **Mathématiques / déterministes** | Seuils, conditions binaires | Code / moteur de règles |
| **Qualitatives / sémantiques** | Jugement, équivalences, contexte | Agent IA + embeddings |
| **Hybrides / mixtes** | Sémantique → puis calcul strict | Routage dynamique |

---

## Catégorie 1 — Règles mathématiques

<span class="tag-math">DÉTERMINISTE</span>

**Exemples**
- Expérience professionnelle totale ≥ 4 ans
- Âge > 21 ans
- Seuils de crédits, fractions d'années de scolarité

**Traitement préféré**
- Boucles / conditions dans du **code** (fonctions, moteur de règles)
- L'IA sert uniquement à **extraire les paramètres** (valeurs numériques, dates) à injecter dans ces fonctions
- Pas d'interprétation sémantique du verdict

---

## Catégorie 2 — Règles qualitatives

<span class="tag-sem">SÉMANTIQUE</span>

**Exemples**
- *"Devrait avoir un background de développement dans un domaine complexe"*
- Équivalences de cursus, similarité de libellés
- Jugement de contexte académique ou professionnel

**Traitement préféré**
- **Agent IA** + **recherche sémantique** (embeddings / base vectorielle)
- Un LLM peut associer un projet de *coordination d'une flotte de drones* à un *domaine complexe* — naturellement aligné avec les embeddings

---

## Catégorie 3 — Règles hybrides

<span class="tag-hybrid">HYBRIDE</span>

**Définition :** interprétation sémantique **puis** condition mathématique stricte

**Exemple**
> Identifier les cours équivalents au domaine X, **puis** vérifier que la somme des crédits ≥ 60

**Traitement préféré**
- Routage dynamique :
  1. Agent extrait / interprète → produit des variables
  2. Variables → moteur de calcul pour la décision finale

---

## Architecture cible

```
Référentiel de règles
        │
        ▼
┌───────────────────┐
│  Labellisation    │  ← Llama hébergé (modèle moyen suffit)
│  (cartographie)   │
└────────┬──────────┘
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
Code / moteur      Agent IA + embeddings
(déterministe)        (sémantique)
    │                     │
    └──────────┬──────────┘
               ▼
        Décision finale
        (Agent #2, dossiers réels)
```

---

## Posture & calibrage

**La labellisation ≠ évaluation de dossiers**
C'est une **étape de cadrage architectural**

<br>

| Phase | Modèle requis | Objectif |
|---|---|---|
| **Labellisation du référentiel** | Modèle moyen (ex. Llama) | Cartographier règle → catégorie |
| **Évaluation de vrais dossiers** | Modèle puissant (Agent #2) | Appliquer les règles à des cas réels |

---

## Livrables attendus

De la boucle de labellisation :

1. **Rapport traçable** — règle → catégorie → justification
2. **Distribution par catégorie** — quelle part du référentiel est déterministe vs. sémantique
3. **File de révision** — cas ambigus soumis à validation humaine

---

<!-- _class: title -->

## Message clé

> Séparer **calcul** et **interprétation** :
> le code pour les seuils rigides,
> l'agent pour le sémantique —
> et **labelliser d'abord** pour savoir quelle part
> du référentiel relève de chaque monde.
