#!/usr/bin/env python3
"""
Analyseur Statistique CSV - Outil d'analyse automatique de données
Auteur: Nono271105
Version: 1.0

Cet outil réalise une analyse statistique complète d'un fichier CSV :
- Statistiques descriptives
- Matrice de corrélation 
- Analyses de régression
- Visualisations automatiques
- Rapport détaillé avec recommandations

Usage:
    python csv_analyzer.py fichier.csv
    python csv_analyzer.py fichier.csv --output resultats/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import argparse
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CSVAnalyzer:
    """Analyseur statistique pour fichiers CSV"""
    
    def __init__(self, filepath):
        """
        Initialise l'analyseur avec un fichier CSV
        
        Args:
            filepath (str): Chemin vers le fichier CSV
        """
        self.filepath = filepath
        self.data = None
        self.numeric_columns = []
        self.stats_summary = {}
        self.correlation_matrix = None
        self.strong_correlations = []
        self.regression_results = {}
        self.insights = []
        self.recommendations = []
        
    def load_data(self):
        """Charge et prépare les données CSV"""
        try:
            # Tentative de lecture avec différents séparateurs
            separators = [',', ';', '\t', '|']
            self.data = None
            
            for sep in separators:
                try:
                    self.data = pd.read_csv(self.filepath, sep=sep, encoding='utf-8')
                    if self.data.shape[1] > 1:  # Plus d'une colonne = bon séparateur
                        break
                except:
                    continue
            
            if self.data is None:
                # Dernière tentative avec détection automatique
                self.data = pd.read_csv(self.filepath, encoding='utf-8')
                
            print(f"✓ Données chargées: {self.data.shape[0]} lignes, {self.data.shape[1]} colonnes")
            
            # Identifier les colonnes numériques
            self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            print(f"✓ Colonnes numériques identifiées: {len(self.numeric_columns)}")
            
            if len(self.numeric_columns) < 2:
                print("⚠️  Attention: Moins de 2 colonnes numériques détectées.")
                print("   L'analyse de corrélation sera limitée.")
                
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            sys.exit(1)
    
    def calculate_descriptive_stats(self):
        """Calcule les statistiques descriptives"""
        print("\n📊 Calcul des statistiques descriptives...")
        
        numeric_data = self.data[self.numeric_columns]
        
        self.stats_summary = {
            'Nombre': numeric_data.count(),
            'Moyenne': numeric_data.mean(),
            'Écart-type': numeric_data.std(),
            'Variance': numeric_data.var(),
            'Min': numeric_data.min(),
            'Max': numeric_data.max(),
            'Mediane': numeric_data.median(),
            'q25': numeric_data.quantile(0.25),
            'q75': numeric_data.quantile(0.75),
            'skewness': numeric_data.skew(),
            'kurtosis': numeric_data.kurtosis()
        }
        
        # Détection des valeurs aberrantes (méthode IQR)
        self.outliers_info = {}
        for col in self.numeric_columns:
            Q1 = self.stats_summary['q25'][col]
            Q3 = self.stats_summary['q75'][col]
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            self.outliers_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.data)) * 100,
                'bounds': (lower_bound, upper_bound)
            }
    
    def calculate_correlation_matrix(self):
        """Calcule la matrice de corrélation"""
        print("\n🔗 Calcul de la matrice de corrélation...")
        
        if len(self.numeric_columns) < 2:
            print("⚠️  Impossible de calculer les corrélations: pas assez de variables numériques")
            return
        
        numeric_data = self.data[self.numeric_columns]
        
        # Matrice de corrélation de Pearson
        self.correlation_matrix = numeric_data.corr()
        
        # Identifier les corrélations fortes
        self.strong_correlations = []
        threshold_strong = 0.7
        threshold_moderate = 0.5
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                var1 = self.correlation_matrix.columns[i]
                var2 = self.correlation_matrix.columns[j]
                corr_value = self.correlation_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold_strong:
                    strength = "très forte"
                elif abs(corr_value) >= threshold_moderate:
                    strength = "modérée"
                else:
                    continue
                
                direction = "positive" if corr_value > 0 else "négative"
                
                self.strong_correlations.append({
                    'var1': var1,
                    'var2': var2,
                    'correlation': corr_value,
                    'strength': strength,
                    'direction': direction
                })
        
        # Trier par force de corrélation
        self.strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    def perform_regression_analysis(self):
        """Réalise des analyses de régression linéaire"""
        print("\n📈 Analyse de régression linéaire...")
        
        if not self.strong_correlations:
            print("⚠️  Aucune corrélation significative pour l'analyse de régression")
            return
        
        self.regression_results = {}
        
        for corr_info in self.strong_correlations[:5]:  # Top 5 corrélations
            var1, var2 = corr_info['var1'], corr_info['var2']
            
            # Préparer les données (supprimer les NaN)
            data_clean = self.data[[var1, var2]].dropna()
            
            if len(data_clean) < 10:  # Pas assez de données
                continue
            
            X = data_clean[var1].values.reshape(-1, 1)
            y = data_clean[var2].values
            
            # Régression linéaire
            model = LinearRegression()
            model.fit(X, y)
            
            # Prédictions et métriques
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            # Test de significativité
            n = len(data_clean)
            t_stat = corr_info['correlation'] * np.sqrt((n-2)/(1-corr_info['correlation']**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
            
            self.regression_results[f"{var1}_vs_{var2}"] = {
                'var_x': var1,
                'var_y': var2,
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'r2': r2,
                'correlation': corr_info['correlation'],
                'p_value': p_value,
                'n_samples': n,
                'equation': f"{var2} = {model.coef_[0]:.4f} * {var1} + {model.intercept_:.4f}"
            }
    
    def generate_insights(self):
        """Génère les insights et recommandations"""
        print("\n🧠 Génération des insights...")
        
        self.insights = []
        self.recommendations = []
        
        # Insights sur les statistiques descriptives
        for col in self.numeric_columns:
            skew = self.stats_summary['skewness'][col]
            kurt = self.stats_summary['kurtosis'][col]
            outlier_pct = self.outliers_info[col]['percentage']
            
            if abs(skew) > 1:
                self.insights.append(f"📊 {col}: Distribution asymétrique ({'à droite' if skew > 0 else 'à gauche'})")
            
            if kurt > 3:
                self.insights.append(f"📊 {col}: Distribution leptokurtique (queues lourdes)")
            elif kurt < -1:
                self.insights.append(f"📊 {col}: Distribution platykurtique (queues légères)")
            
            if outlier_pct > 5:
                self.insights.append(f"⚠️  {col}: {outlier_pct:.1f}% de valeurs aberrantes détectées")
        
        # Insights sur les corrélations
        if self.strong_correlations:
            strongest = self.strong_correlations[0]
            self.insights.append(f"🔗 Corrélation la plus forte: {strongest['var1']} ↔ {strongest['var2']} (r={strongest['correlation']:.3f})")
            
            # Détecter la multicolinéarité
            high_corr_count = len([c for c in self.strong_correlations if abs(c['correlation']) > 0.8])
            if high_corr_count > 0:
                self.insights.append(f"⚠️  Risque de multicolinéarité détecté ({high_corr_count} corrélations > 0.8)")
        
        # Insights sur les régressions
        if self.regression_results:
            best_r2 = max(self.regression_results.values(), key=lambda x: x['r2'])
            self.insights.append(f"📈 Meilleure régression: {best_r2['var_x']} → {best_r2['var_y']} (R²={best_r2['r2']:.3f})")
        
        # Recommandations
        if len(self.numeric_columns) >= 3:
            self.recommendations.append("🎯 Explorer l'analyse en composantes principales (PCA)")
            self.recommendations.append("🎯 Considérer des modèles de régression multiple")
        
        if any(abs(c['correlation']) > 0.9 for c in self.strong_correlations):
            self.recommendations.append("⚠️  Attention aux variables redondantes dans les modèles")
        
        if any(self.outliers_info[col]['percentage'] > 10 for col in self.numeric_columns):
            self.recommendations.append("🧹 Évaluer le traitement des valeurs aberrantes")
        
        if any(abs(self.stats_summary['skewness'][col]) > 2 for col in self.numeric_columns):
            self.recommendations.append("🔄 Considérer des transformations (log, Box-Cox)")
    
    def create_visualizations(self, output_dir="output"):
        """Crée les visualisations automatiques"""
        print("\n📊 Génération des visualisations...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Style des graphiques
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Heatmap de corrélation
        if self.correlation_matrix is not None:
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
            sns.heatmap(self.correlation_matrix, 
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       mask=mask,
                       square=True,
                       fmt='.2f',
                       cbar_kws={'label': 'Corrélation de Pearson'})
            plt.title('Matrice de Corrélation', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Graphiques de régression pour les top corrélations
        if self.regression_results:
            n_plots = min(4, len(self.regression_results))
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, (key, result) in enumerate(list(self.regression_results.items())[:n_plots]):
                ax = axes[i]
                
                # Données pour le graphique
                data_clean = self.data[[result['var_x'], result['var_y']]].dropna()
                x = data_clean[result['var_x']]
                y = data_clean[result['var_y']]
                
                # Nuage de points
                ax.scatter(x, y, alpha=0.6, s=30)
                
                # Droite de régression
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = result['slope'] * x_line + result['intercept']
                ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'R² = {result["r2"]:.3f}')
                
                ax.set_xlabel(result['var_x'])
                ax.set_ylabel(result['var_y'])
                ax.set_title(f'Régression: {result["var_x"]} → {result["var_y"]}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Cacher les subplots inutilisés
            for i in range(n_plots, 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/regression_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Histogrammes des variables numériques
        if self.numeric_columns:
            n_cols = min(4, len(self.numeric_columns))
            n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(self.numeric_columns):
                ax = axes[i] if len(self.numeric_columns) > 1 else axes
                
                self.data[col].hist(bins=30, alpha=0.7, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f'Distribution: {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Fréquence')
                ax.grid(True, alpha=0.3)
                
                # Ajouter des statistiques
                mean_val = self.stats_summary['mean'][col]
                median_val = self.stats_summary['median'][col]
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Moyenne: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', label=f'Médiane: {median_val:.2f}')
                ax.legend()
            
            # Cacher les subplots inutilisés
            for i in range(len(self.numeric_columns), n_rows * n_cols):
                if i < len(axes):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Visualisations sauvegardées dans {output_dir}/")
    
    def generate_report(self, output_dir="output"):
        """Génère un rapport complet"""
        print("\n📄 Génération du rapport...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# 📊 RAPPORT D'ANALYSE STATISTIQUE AUTOMATIQUE
**Fichier analysé:** {os.path.basename(self.filepath)}  
**Date d'analyse:** {timestamp}  
**Générateur:** Analyseur Statistique CSV v1.0

---

## 📋 RÉSUMÉ DES DONNÉES
- **Nombre de lignes:** {self.data.shape[0]:,}
- **Nombre de colonnes:** {self.data.shape[1]}
- **Variables numériques:** {len(self.numeric_columns)}
- **Variables analysées:** {', '.join(self.numeric_columns)}

---

## 📊 STATISTIQUES DESCRIPTIVES

"""
        
        # Tableau des statistiques
        if self.numeric_columns:
            stats_df = pd.DataFrame({
                'Variable': self.numeric_columns,
                'Moyenne': [f"{self.stats_summary['mean'][col]:.4f}" for col in self.numeric_columns],
                'Écart-type': [f"{self.stats_summary['std'][col]:.4f}" for col in self.numeric_columns],
                'Médiane': [f"{self.stats_summary['median'][col]:.4f}" for col in self.numeric_columns],
                'Min': [f"{self.stats_summary['min'][col]:.4f}" for col in self.numeric_columns],
                'Max': [f"{self.stats_summary['max'][col]:.4f}" for col in self.numeric_columns],
                'Valeurs aberrantes (%)': [f"{self.outliers_info[col]['percentage']:.1f}%" for col in self.numeric_columns]
            })
            
            report += stats_df.to_string(index=False)
        
        report += f"""

---

## 🔗 ANALYSE DE CORRÉLATION

"""
        
        if self.strong_correlations:
            report += "### Corrélations significatives détectées:\n\n"
            for corr in self.strong_correlations:
                report += f"- **{corr['var1']} ↔ {corr['var2']}:** r = {corr['correlation']:.3f} ({corr['strength']} {corr['direction']})\n"
        else:
            report += "Aucune corrélation significative détectée (seuil: |r| > 0.5)\n"
        
        report += f"""

---

## 📈 ANALYSE DE RÉGRESSION

"""
        
        if self.regression_results:
            report += "### Modèles de régression linéaire:\n\n"
            for key, result in self.regression_results.items():
                significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                report += f"""
**{result['var_x']} → {result['var_y']}** {significance}
- Équation: {result['equation']}
- R² = {result['r2']:.4f} ({result['r2']*100:.1f}% de variance expliquée)
- Corrélation: r = {result['correlation']:.3f}
- p-value: {result['p_value']:.6f}
- Échantillon: {result['n_samples']} observations

"""
        else:
            report += "Aucune analyse de régression réalisée.\n"
        
        report += f"""

---

## 🧠 INSIGHTS PRINCIPAUX

"""
        
        if self.insights:
            for insight in self.insights:
                report += f"- {insight}\n"
        else:
            report += "Aucun insight particulier détecté.\n"
        
        report += f"""

---

## 🎯 RECOMMANDATIONS

"""
        
        if self.recommendations:
            for rec in self.recommendations:
                report += f"- {rec}\n"
        else:
            report += "Aucune recommandation spécifique.\n"
        
        report += f"""

---

## 📊 VISUALISATIONS GÉNÉRÉES

Les graphiques suivants ont été créés automatiquement:
- `correlation_heatmap.png` - Carte de chaleur des corrélations
- `regression_plots.png` - Graphiques de régression (top 4)
- `distributions.png` - Histogrammes des variables

---

## 🔍 INTERPRÉTATION DES RÉSULTATS

### Corrélations:
- **|r| > 0.8** : Corrélation très forte (attention multicolinéarité)
- **|r| > 0.6** : Corrélation forte
- **|r| > 0.4** : Corrélation modérée
- **|r| < 0.2** : Corrélation faible

### Régression:
- **R² > 0.8** : Excellent pouvoir prédictif
- **R² > 0.6** : Bon pouvoir prédictif
- **R² > 0.4** : Pouvoir prédictif modéré
- **R² < 0.2** : Pouvoir prédictif faible

### Significativité:
- **p < 0.001** : Très hautement significatif (***)
- **p < 0.01** : Hautement significatif (**)
- **p < 0.05** : Significatif (*)

---

*Rapport généré automatiquement par l'Analyseur Statistique CSV*
"""
        
        # Sauvegarder le rapport
        with open(f"{output_dir}/rapport_analyse.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"✓ Rapport sauvegardé: {output_dir}/rapport_analyse.md")
        
        return report
    
    def run_complete_analysis(self, output_dir="output"):
        """Exécute l'analyse complète"""
        print(f"\n🚀 DÉBUT DE L'ANALYSE COMPLÈTE")
        print(f"📁 Fichier: {self.filepath}")
        print(f"📁 Sortie: {output_dir}/")
        print("=" * 50)
        
        # Étapes de l'analyse
        self.load_data()
        self.calculate_descriptive_stats()
        self.calculate_correlation_matrix()
        self.perform_regression_analysis()
        self.generate_insights()
        self.create_visualizations(output_dir)
        report = self.generate_report(output_dir)
        
        print("\n" + "=" * 50)
        print("✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
        print(f"📊 {len(self.numeric_columns)} variables analysées")
        print(f"🔗 {len(self.strong_correlations)} corrélations significatives")
        print(f"📈 {len(self.regression_results)} modèles de régression")
        print(f"🧠 {len(self.insights)} insights générés")
        print(f"📁 Résultats dans: {output_dir}/")
        
        return report

def main():
    """Fonction principale pour l'interface en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Analyseur Statistique CSV - Analyse automatique de données",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python csv_analyzer.py data.csv
  python csv_analyzer.py data.csv --output results/
  python csv_analyzer.py data.csv --output analysis/ --show-report
        """
    )
    
    parser.add_argument("filepath", help="Chemin vers le fichier CSV à analyser")
    parser.add_argument("--output", "-o", default="output", 
                       help="Répertoire de sortie (défaut: output)")
    parser.add_argument("--show-report", action="store_true",
                       help="Afficher le rapport dans le terminal")
    parser.add_argument("--version", action="version", version="CSV Analyzer 1.0")
    
    args = parser.parse_args()
    
    # Vérifier que le fichier existe
    if not os.path.exists(args.filepath):
        print(f"❌ Erreur: Le fichier '{args.filepath}' n'existe pas.")
        sys.exit(1)
    
    # Créer l'analyseur et lancer l'analyse
    analyzer = CSVAnalyzer(args.filepath)
    report = analyzer.run_complete_analysis(args.output)
    
    # Afficher le rapport si demandé
    if args.show_report:
        print("\n" + "=" * 60)
        print("📄 RAPPORT D'ANALYSE")
        print("=" * 60)
        print(report)

if __name__ == "__main__":
    main()