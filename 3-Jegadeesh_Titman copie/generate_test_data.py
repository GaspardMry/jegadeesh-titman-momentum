# GÉNÉRATEUR DE DONNÉES DE TEST — Simuler le fichier CRSP
# Création d'un fichier CSV "crsp_data_test.csv" avec la même structure
# que les vraies données CRSP (1965-1989).
# =============================================================================

import pandas as pd
import numpy as np

np.random.seed(42)   # seed fixe pour résultats reproductibles

# Paramètres
N_TITRES    = 500    # nombre de titres dans l'univers
N_SEMESTRES = 50     # de Semester=1 à Semester=50 (ce qui vaut environ à 25 ans)

# Bourses : ~70% NYSE, ~30% AMEX (+ quelques NASDAQ/autres à filtrer)
BOURSES = ['N'] * 700 + ['A'] * 250 + ['Q'] * 40 + ['X'] * 10

rows = []

for permno in range(1, N_TITRES + 1):
    # Chaque titre est coté sur une bourse tirée au sort (une fois pour tout)
    bourse = np.random.choice(BOURSES)
    # Paramètres de rentabilité propres à ce titre
    # (certains titres sont globalement plus performants que d'autres)
    drift_propre = np.random.normal(0, 0.005)
    for semester in range(1, N_SEMESTRES + 1):
        # Chaque semestre = 6 mois → 6 lignes par titre par semestre
        for mois in range(6):
            # Rentabilité mensuelle ~ normale(drift, volatilité)
            # On ajoute un effet momentum : les bons titres ont tendance
            # à rester bons (autocorrélation positive légère)
            ret = np.random.normal(drift_propre + 0.005, 0.08)
            # Date fictive (le format exact n'est pas utilisé dans notre code)
            date = 19650101 + (semester - 1) * 600 + mois * 100
            # Parfois, données manquantes (~2% des cas)
            if np.random.random() < 0.02:
                ret = np.nan
            rows.append({
                'PERMNO'   : permno,
                'date'     : date,
                'PRIMEXCH' : bourse,
                'RET'      : ret,
                'Semester' : semester
            })
df = pd.DataFrame(rows)

print(f"Données générées : {len(df):,} lignes")
print(f"Titres           : {df['PERMNO'].nunique()}")
print(f"Semestres        : {df['Semester'].min()} → {df['Semester'].max()}")
print(f"Bourses          : {df['PRIMEXCH'].value_counts().to_dict()}")
print(f"Valeurs NaN      : {df['RET'].isna().sum()}")

output_path = "crsp_data_test.csv"
df.to_csv(output_path, index=False)
print(f"\nFichier sauvegardé : {output_path}")
print("Tu peux maintenant lancer momentum_strategy.py")
