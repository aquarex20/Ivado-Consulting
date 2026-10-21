# Contexte — Labellisation des règles pour choisir l’architecture

## Objectif du système

Nous construisons un système d’IA capable d’émettre une **décision** à partir de :

1. **règles prédéterminées** (référentiel métier) ;
2. **dossiers de candidature** (documents, parcours, critères).

L’enjeu n’est pas seulement « que le modèle raisonne bien », mais d’adopter une **architecture performante** : réduire la complexité sous-jacente lorsque un agent doit retrouver le contexte de règles applicable à une tâche.

## Pourquoi labelliser les règles d’abord

Confier **toutes** les règles à un seul agent (recherche + interprétation + verdict) augmente le bruit, le coût et le risque d’erreur.

Avant de figer la structure d’exécution, il faut donc **labelliser le référentiel** : pour chaque règle (ou sous-règle), savoir à quelle **catégorie** elle appartient. Cette cartographie oriente ensuite le choix d’architecture.

## Catégories et traitement idéal

### 1. Règles mathématiques / déterministes

**Exemples :**

- *expérience professionnelle totale ≥ 4 ans* ;
- *âge > 21 ans* ;
- seuils de crédits, fractions d’années de scolarité, conditions binaires claires.

**Traitement préféré :**

- boucles / conditions dans du **code** (fonctions, moteur de règles) ;
- l’IA, si elle intervient, sert surtout à **extraire les paramètres** (valeurs numériques, dates, indicateurs) à injecter dans ces fonctions ;
- pas d’interprétation sémantique du verdict lui-même.

### 2. Règles qualitatives / sémantiques

**Exemples :**

- *devrait avoir un background de développement dans un domaine complexe* ;
- équivalences de cursus, similarité de libellés, jugement de contexte académique ou professionnel.

**Traitement préféré :**

- **agent IA** + **recherche sémantique** (embeddings / base vectorielle) ;
- la dimension linguistique colle naturellement aux embeddings : un LLM peut associer, par exemple, un projet de *coordination d’une flotte de drones* à un *domaine complexe*.

### 3. Règles hybrides / mixtes (si applicable)

**Définition :** interprétation sémantique **puis** condition mathématique stricte  
(ex. identifier les cours équivalents au domaine X, **puis** vérifier que la somme des crédits ≥ 60).

**Traitement préféré :**

- routage dynamique : l’agent extrait / interprète → variables → moteur de calcul.

## Posture pour la présentation

- La labellisation n’est **pas** encore l’évaluation de dossiers ; c’est une **étape de cadrage architectural**.
- Un modèle moyen (ex. Llama hébergé) peut suffire pour **cartographier** le référentiel ; l’exigence de puissance monte surtout quand on applique les règles à de vrais dossiers (Agent #2).
- Livrable attendu de la boucle de labellisation : rapport traçable (règle → catégorie → justification), distribution par catégorie, et file de révision pour les cas ambigus.

## Message clé

> Séparer **calcul** et **interprétation** : le code pour les seuils rigides, l’agent pour le sémantique — et labelliser d’abord pour savoir quelle part du référentiel relève de chaque monde.
