# ML Disaster Survival Project

Prédiction de survie multi-catastrophes par apprentissage automatique sur trois naufrages maritimes historiques : le Titanic (1912), la Lusitania (1915) et l'Estonia (1994).

---

## Présentation du projet

Ce projet construit et évalue des modèles de prédiction de survie sur des données combinées de trois catastrophes maritimes. Le défi central est la généralisation inter-domaine sous décalage de distribution : chaque catastrophe présente une disponibilité de données différente, des mécanismes de survie différents et des conventions d'enregistrement historiques différentes.

Le projet couvre l'intégralité du pipeline ML : collecte de données (dont scraping web), ingénierie des variables avec harmonisation entre sources hétérogènes, visualisation exploratoire, entraînement de modèles avec deux stratégies concurrentes (global vs séparé), analyse de l'importance des variables par quatre méthodes, test de généralisation sur un holdout, et une analyse complète des biais et des implications éthiques.

---

## Structure du dépôt

```
ml-disaster-survival-project/
│
├── data/
│   ├── external/                  # Jeux de données téléchargés
│   │   ├── Titanic.csv            # Dataset Kaggle Titanic
│   │   └── estonia-passenger-list.csv  # Dataset Kaggle Estonia
│   ├── raw/
│   │   └── lusitania_people_1915_all_tables.csv  # Données Lusitania scrapées
│   └── processed/
│       ├── disaster_ml_dataset.csv     # Dataset unifié prêt pour le ML
│       └── disaster_viz_dataset.csv    # Version MultiIndex pour la visualisation
│
├── notebooks/
│   ├── 01_data_collection.ipynb      # Chargement, scraping, EDA
│   ├── 02_Feature_Engineering.ipynb  # Nettoyage, création de variables, harmonisation
│   ├── 03_Data_Viz.ipynb             # Visualisation exploratoire multi-catastrophes
│   ├── 04_models.ipynb               # Entraînement, importance des variables, SHAP
│   └── 05_evaluation.ipynb           # Holdout, analyse des biais, éthique
│
├── models/
│   ├── best_pooled_model.pkl      # Modèle GradBoost global sauvegardé
│   ├── titanic_model.pkl          # Modèle spécifique Titanic sauvegardé
│   ├── pooled_split.pkl           # Découpages train/test stratégie globale
│   └── titanic_split.pkl          # Découpages train/test stratégie Titanic
│
├── reports/
│   ├── data_collection_report.md  # Sources, qualité des données, considérations éthiques
│   └── technical_report.md        # Méthodologie complète et résultats
│
├── requirements.txt
└── README.md
```

---

## Comment lancer le projet

### 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/ml-disaster-survival-project.git
cd ml-disaster-survival-project
```

### 2. Créer et activer l'environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer les notebooks dans l'ordre

```bash
jupyter notebook
```

Ouvrir et exécuter les notebooks dans l'ordre suivant :

| Notebook | Description |
|----------|-------------|
| `01_data_collection.ipynb` | Chargement des CSV Titanic et Estonia, scraping de la Lusitania depuis rmslusitania.info, EDA |
| `02_Feature_Engineering.ipynb` | Nettoyage, ingénierie des variables, harmonisation des schémas, construction du dataset unifié |
| `03_Data_Viz.ipynb` | Visualisation exploratoire : taux de survie par sexe, classe, âge, groupe, nationalité et tarif — par catastrophe et en vue agrégée (heatmap, histogrammes, barplots) |
| `04_models.ipynb` | Entraînement des modèles global et séparés, analyse de l'importance des variables (permutation, SHAP, coefficients, column-drop) |
| `05_evaluation.ipynb` | Généralisation holdout sur la Lusitania, analyse du décalage de covariance, analyse des biais, discussion éthique |

**Note** : Les notebooks utilisent des chemins absolus à mettre à jour selon votre structure de répertoires locale. Rechercher `C:/Users/noura/` et remplacer par le chemin de votre projet.

---

## Résultats principaux

### Performances des modèles

| Stratégie | Modèle | ROC-AUC | F1 | Seuil |
|-----------|--------|---------|-----|-------|
| Global | GradBoost (réduit, 7 variables) | 0,757 | 0,597 | 0,35 |
| Séparé — Titanic | GradBoost (réduit, 6 variables) | 0,860 | 0,776 | 0,50 |
| Séparé — Estonia | SVM | 0,742 | 0,396 | 0,50 |
| Séparé — Lusitania | RandomForest | ~0,55 | ~0,45 | 0,50 |
| Holdout (Lusitania) | Modèle global | **0,526** | — | — |

### Importance des variables (modèle global)

Meilleures variables par importance par permutation (chute de ROC-AUC) :

1. `age_group` — 0,021
2. `Sex` — 0,019
3. `Fare` — 0,012
4. `group_size` — 0,003

Variables à impact quasi-nul ou négatif (exclues du modèle final) : `nationality_region`, `night`, `wartime`, `ship_size`, `has_group`.

### Analyse des biais

Résultats du Disparate Impact Ratio (DIR) — valeurs inférieures à 0,8 indiquent une discrimination potentielle :

| Dimension | Groupe le plus défavorisé | DIR |
|-----------|--------------------------|-----|
| Genre | Homme | 0,355 |
| Âge | Senior | 0,070 |
| Classe de voyage | Passager (Estonia) | 0,014 |
| Nationalité | nordic_baltic | 0,091 |

Le modèle amplifie les inégalités historiques existantes au-delà des données. Ceci est documenté et contextualisé plutôt que corrigé, car le modèle est utilisé uniquement pour l'analyse historique — pas pour la prise de décision.

---

## Déclaration éthique

Ce projet utilise des données personnelles de vraies victimes de catastrophes maritimes à des fins de recherche académique uniquement. Toutes les données ont été obtenues depuis des sources publiquement accessibles (Kaggle, dossiers d'enquête officiels, archives historiques publiques). L'analyse des biais révèle que le modèle apprend des patterns de survie historiquement discriminatoires — cela est attendu compte tenu des données, et est explicitement documenté plutôt que dissimulé. Le modèle ne doit pas être utilisé à des fins décisionnelles.

Voir `reports/data_collection_report.md` pour les considérations éthiques complètes et `reports/technical_report.md` Section 6 pour la discussion du paradoxe éthique.

---

## Dépendances

Voir `requirements.txt` pour la liste complète. Packages principaux :

- `scikit-learn >= 1.3`
- `pandas >= 2.0`
- `numpy`
- `matplotlib`
- `shap`
- `joblib`
- `requests` + `beautifulsoup4` (pour le scraping de la Lusitania)
- `missingno` (pour la visualisation des données manquantes)
- `jupyter`

---

## Sources des données

| Jeu de données | Source | URL |
|---------------|--------|-----|
| Titanic | Kaggle | https://www.kaggle.com/competitions/titanic |
| Estonia | Kaggle (Stanford) | https://www.kaggle.com/datasets/christianlillelund/passenger-list-for-the-estonia-ferry-disaster |
| Lusitania | rmslusitania.info (scraping) | https://www.rmslusitania.info/people/ |
