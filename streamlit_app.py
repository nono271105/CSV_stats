"""
Interface Streamlit pour l'Analyseur Statistique CSV
Version: 1.0

Interface web conviviale pour l'analyse automatique de données CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
from datetime import datetime

# Tenter d'importer le module principal
try:
    from csv_analyzer import CSVAnalyzer
except ImportError:
    st.error("❌ Module `csv_analyzer.py` non trouvé. Assurez-vous qu'il est dans le même répertoire que ce fichier.")
    st.stop()

# Configuration de la page
st.set_page_config(
    page_title="Analyseur Statistique CSV",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.insight-box {
    background-color: #e8f4ff;
    padding: 1rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
    border-radius: 0.25rem;
}

.warning-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-left: 4px solid #ffc107;
    margin: 0.5rem 0;
    border-radius: 0.25rem;
}

.success-box {
    background-color: #d4edda;
    padding: 1rem;
    border-left: 4px solid #28a745;
    margin: 0.5rem 0;
    border-radius: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

def display_results(analyzer):
    """Affichage des résultats dans Streamlit"""
    
    st.subheader("📈 Statistiques descriptives")
    # Correction ici: utiliser analyzer.stats_summary qui est le dictionnaire des stats
    # et le convertir en DataFrame pour un affichage propre
    if analyzer.stats_summary:
        stats_df = pd.DataFrame(analyzer.stats_summary).T # Transposer pour avoir les variables en colonnes
        st.dataframe(stats_df)
    else:
        st.info("Aucune statistique descriptive à afficher (vérifiez les colonnes numériques).")


    st.subheader("🔗 Matrice de corrélation")
    if analyzer.correlation_matrix is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(analyzer.correlation_matrix, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        st.pyplot(fig)
    else:
        st.info("Aucune matrice de corrélation à afficher (pas assez de variables numériques).")


    st.subheader("📊 Régressions détectées")
    if analyzer.regression_results:
        for key, result in analyzer.regression_results.items():
            st.markdown(f"**{key}**")
            # Afficher les résultats de manière plus lisible
            st.write(f"- Équation: `{result['equation']}`")
            st.write(f"- R²: `{result['r2']:.4f}`")
            st.write(f"- Corrélation: `{result['correlation']:.3f}`")
            st.write(f"- p-value: `{result['p_value']:.6f}`")
            st.write(f"- Échantillon: `{result['n_samples']}` observations")
            st.markdown("---")
    else:
        st.info("Aucune régression significative détectée.")

    st.subheader("💡 Insights")
    if analyzer.insights:
        for insight in analyzer.insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    else:
        st.info("Aucun insight généré.")

def main():
    """Fonction principale de l'interface Streamlit"""
    
    st.markdown('<h1 class="main-header">📊 Analyseur Statistique CSV</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("**Version:** 1.0")
    st.sidebar.markdown("**Auteur:** Nono271105")
    
    uploaded_file = st.file_uploader(
        "📁 Choisissez un fichier CSV à analyser",
        type=['csv'],
        help="Formats supportés: CSV avec séparateurs , ; | ou tabulation"
    )
    
    if uploaded_file is not None:
        # Sauvegarde temporaire du fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner("🔄 Chargement et analyse des données..."):
                analyzer = CSVAnalyzer(tmp_file_path)
                analyzer.load_data()
                analyzer.calculate_descriptive_stats() # Cette méthode calcule les stats et les stocke dans stats_summary
                analyzer.calculate_correlation_matrix()
                analyzer.perform_regression_analysis()
                analyzer.generate_insights()

            display_results(analyzer)
        
        except Exception as e:
            st.error(f"❌ Une erreur s'est produite lors de l'analyse : {e}")
            st.exception(e) # Affiche la trace complète de l'erreur pour le débogage

        finally:
            # Nettoyage du fichier temporaire
            os.remove(tmp_file_path)

if __name__ == "__main__":
    main()
