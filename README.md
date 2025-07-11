# Analyseur Statistique CSV 📊
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Outil d'analyse automatique de fichiers CSV avec interface web Streamlit.

## Fonctionnalités

- Statistiques descriptives automatiques
- Matrice de corrélation
- Analyses de régression linéaire
- Visualisations (heatmap, histogrammes, régressions)
- Génération d'un rapport détaillé avec recommandations
- Interface web conviviale via Streamlit

## Installation

1. Clonez ce dépôt ou copiez les fichiers dans un dossier.
2. Installez les dépendances Python :

```sh
pip install -r requierments.txt
```

## Utilisation avec l'interface web Streamlit

Lancez l'application web :

```sh
streamlit run streamlit_app.py
```

Puis ouvrez le lien affiché dans votre navigateur. Chargez un fichier CSV pour obtenir l'analyse interactive.

## Fichiers du projet

- `csv_analyzer.py` : Module principal d'analyse statistique
- `streamlit_app.py` : Interface web Streamlit
- `MOCK_DATA.csv` : Exemple de données CSV
- `requierments.txt` : Liste des dépendances Python

