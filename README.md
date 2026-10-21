# Pipeline de classification des règles ICARE

Outil interne qui parcourt le corpus ICARE (règles générales `G*` et particulières `P*`),
extrait le texte, segmente chaque fichier en sous-règles, puis interroge Azure OpenAI
pour étiqueter chaque sous-règle dans l'une des trois catégories :

- **Déterministe** — calcul strict / seuil / logique binaire.
- **Qualitative** — interprétation sémantique / équivalences.
- **Hybride** — extraction sémantique **puis** calcul / seuil.

Voir [`config/classification_prompt.md`](config/classification_prompt.md) pour la définition
détaillée, et le [plan complet](.cursor/plans/) pour l'architecture cible.

## Installation

Sur macOS, si le projet est stocké dans un dossier synchronisé
(OneDrive, iCloud…), créer le venv **en dehors** de ce dossier pour éviter que
l'attribut étendu `com.apple.provenance` ne force le drapeau `hidden` sur les
fichiers `.pth` de l'editable install (Python les ignore alors silencieusement).

```bash
python3 -m venv ~/.venvs/icare-rules
source ~/.venvs/icare-rules/bin/activate
pip install -e ".[dev]"

# Environ deux tiers des `.doc` du corpus sont en fait des fichiers WordPerfect :
# `textutil` n'en tire que du binaire, il faut `wpd2text`.
brew install libwpd
```

## Configuration

1. Copier `.env.example` vers `.env` et renseigner les identifiants Azure OpenAI.
2. Copier `config/settings.example.yaml` vers `config/settings.yaml` et adapter au besoin.

## Utilisation

Toutes les commandes s'exécutent via l'entrée `icare-rules` :

```bash
# 1. Inventaire du corpus (dédoublonne .doc/.docx)
#    `--no-hash` recommandé pour un premier passage sur un dossier OneDrive/iCloud.
icare-rules inventory --no-hash --workers 16

# 2. Extraction du texte + segmentation en sous-règles
#    (`--limit N` pour tester sur un sous-ensemble)
icare-rules extract --limit 30

# 2b. Profil du corpus sans LLM (quantités accordées, renvois, pièces exigées,
#     gabarits, mots-clés TF-IDF) → rule_profile.{jsonl,csv}, rule_references.csv,
#     corpus_profile.json
icare-rules profile

# 3a. Dry-run heuristique local (sans Azure) pour valider la boucle
icare-rules classify --dry-run

# 3b. Pilote stratifié : classe un échantillon G/P via Azure OpenAI
icare-rules pilot

# 4. Traitement complet du corpus
#    `--shuffle` : ordre aléatoire reproductible, les résultats partiels restent représentatifs.
#    En local : `backend: ollama` + `deployment: "llama3.2"` dans settings.yaml
#    (ou `--backend ollama --model llama3.2`). Reprise automatique après interruption.
icare-rules classify --shuffle

# 5. Génération des rapports (CSV détaillé, synthèse, révision, métriques)
icare-rules report

# Alternative : chaîner toutes les étapes
icare-rules run --limit 50
```

## PoC retrieval (Qdrant + Streamlit)

Petit sous-ensemble (25 règles) indexé dans Qdrant local, avec trois agents
visibles dans Streamlit : recherche hybride → ligne exacte du PDF → reformulation.

```bash
pip install -e ".[dev]"
icare-poc ingest --max-rules 25
icare-poc ui
# ou : streamlit run app_poc.py
```

Les PDF immuables, le Markdown et l'index sont écrits sous `corpus/`.

Les artefacts sont écrits dans `data/output/` ; le cache d'appels LLM (par hash +
modèle + version du prompt) est écrit dans `data/cache/`. Les documents sources
ne sont jamais modifiés.

## Traçabilité

Chaque sous-règle produit une ligne JSONL avec :
- identifiant stable (`G0539`, `P0048#003`),
- chemin source et empreinte SHA-256 du fichier,
- extrait normalisé,
- catégorie, justification, indices textuels, confiance, drapeau `manual_review`,
- attributs descriptifs (prompt v2) : thème, mots-clés, diplômes, établissements,
  pays, prérequis, valeur accordée, variables à extraire du dossier
  (agrégés dans `attribute_vocabulary.csv` pour construire une taxonomie fermée,
  à imposer ensuite via `classification.themes`),
- version du prompt, deployment Azure, timestamp UTC.

## Sécurité

- Aucun dossier de candidat n'est envoyé au modèle : seul le texte des règles du
  Manuel est transmis.
- Le fichier `.env` et le répertoire `data/` sont ignorés par git.
