# =============================================================================
# STRATÉGIE MOMENTUM — Jegadeesh & Titman (1993)
# "Returns to Buying Winners and Selling Losers"
#
# La Logique de la stratégie consiste à :
#   1. Chaque semestre, on regarde les rentabilités des 12 mois précédents
#   2. On classe les actions en 10 déciles selon leur performance cumulée
#   3. On achète les WINNERS (décile 9 = top 10%) et on vend les LOSERS (décile 0)
#   4. On mesure la performance du portefeuille sur les 6 mois suivants
#   5. On itère sur toutes les périodes et on teste si le gain est significatif
# =============================================================================
import pandas as pd
import numpy as np
from scipy import stats

# =============================================================================
# ÉTAPE 0 — CHARGEMENT DES DONNÉES
# =============================================================================
# Il faut remplacer ce chemin par le chemin de ton vrai fichier CSV CRSP
# Exemple : DATA_PATH = "/Users/toi/Downloads/crsp_data.csv"

DATA_PATH = "crsp_data_test.csv"   # fichier de test généré par generate_test_data.py


def charger_donnees(path):
    """
    Charge le fichier CSV et affiche un aperçu.

    Le CSV contient les colonnes :
      - PERMNO    : identifiant unique d'un titre boursier
      - date      : date au format aaaammjj (ex: 19650131)
      - PRIMEXCH  : bourse de cotation (N=NYSE, A=AMEX, Q=NASDAQ...)
      - RET       : rentabilité mensuelle du titre ce mois-là
      - Semester  : numéro de semestre (1 = jan-juin 1965, 2 = juil-déc 1965, ...)
    """
    print("=" * 60)
    print("ÉTAPE 0 — Chargement des données")
    print("=" * 60)

    df = pd.read_csv(path)

    print(f"Lignes chargées     : {len(df):,}")
    print(f"Colonnes            : {list(df.columns)}")
    print(f"Titres uniques      : {df['PERMNO'].nunique():,}")
    print(f"Périodes (Semester) : {df['Semester'].min()} → {df['Semester'].max()}")
    print()
    print("Aperçu des 5 premières lignes :")
    print(df.head())
    print()

    return df


# =============================================================================
# ÉTAPE 1 — NETTOYAGE DES DONNÉES
# =============================================================================

def nettoyer_donnees(df):
    """
    Deux filtres sont appliqués :

    1. Suppression des valeurs manquantes (NaN)
       → pandas représente les données absentes par NaN (Not a Number)
       → dropna() supprime toutes les lignes qui en contiennent

    2. Conservation uniquement des titres cotés sur NYSE (N) ou AMEX (A)
       → isin(['N', 'A']) renvoie True si la valeur est dans la liste
       → on filtre le DataFrame avec ce masque booléen
    """
    print("=" * 60)
    print("ÉTAPE 1 — Nettoyage des données")
    print("=" * 60)

    n_avant = len(df)

    # Suppression des lignes avec des valeurs manquantes
    df = df.dropna()
    print(f"Après dropna()      : {len(df):,} lignes (supprimé {n_avant - len(df):,})")

    # Filtrage sur la bourse : uniquement NYSE et AMEX
    df = df[df['PRIMEXCH'].isin(['N', 'A'])]
    print(f"Après filtre bourse : {len(df):,} lignes")
    print(f"Bourses présentes   : {df['PRIMEXCH'].unique()}")
    print()

    return df


# =============================================================================
# ÉTAPE 2 — CONSTITUTION DES PORTEFEUILLES WINNERS ET LOSERS
# =============================================================================

def constituer_portefeuilles(df, semester_formation):
    """
    Pour une période de FORMATION de 2 semestres consécutifs (= 12 mois) :

    a) On calcule le taux de rentabilité CUMULÉ de chaque titre sur 12 mois
       → groupby('PERMNO') : on regroupe toutes les lignes d'un même titre
       → sum() sur RET : on additionne les rentabilités mensuelles
         (approximation de la rentabilité cumulée — Jegadeesh & Titman utilisent
          une somme simple des log-rendements ou des rendements arithmétiques)

    b) On classe les titres en 10 déciles avec pd.qcut()
       → décile 0 = 10% des titres les moins performants  → LOSERS
       → décile 9 = 10% des titres les plus performants   → WINNERS

    c) On ne garde que les déciles 0 et 9
    """

    # Les 12 mois de formation = les 2 semestres précédant le semestre courant
    # semester_formation   : premier semestre de la période de 12 mois
    # semester_formation+1 : deuxième semestre
    semestres_formation = [semester_formation, semester_formation + 1]

    # Filtrage sur la période de formation
    df_formation = df[df['Semester'].isin(semestres_formation)]

    # ── Calcul des rentabilités cumulées ──────────────────────────────────────
    # groupby('PERMNO') : crée des "sous-tableaux" par titre
    # [['RET']].sum()   : pour chaque groupe, additionne la colonne RET
    rentabilites_cumulees = (
        df_formation
        .groupby('PERMNO')[['RET']]
        .sum()
        .rename(columns={'RET': 'RET_cum'})   # renommage pour éviter les conflits
        .reset_index()
    )

    # ── Création des 10 déciles ───────────────────────────────────────────────
    # pd.qcut divise les données en 10 groupes d'effectifs ÉGAUX
    # labels=False  → retourne 0,1,...,9 au lieu de (0.1, 0.2], etc.
    # duplicates='drop' → évite une erreur si des valeurs sont identiques
    rentabilites_cumulees['decile'] = pd.qcut(
        rentabilites_cumulees['RET_cum'],
        q=10,
        labels=False,
        duplicates='drop'
    )

    # ── Conservation des déciles extrêmes uniquement ─────────────────────────
    portefeuilles = rentabilites_cumulees[
        rentabilites_cumulees['decile'].isin([0, 9])
    ][['PERMNO', 'decile']]

    n_losers  = (portefeuilles['decile'] == 0).sum()
    n_winners = (portefeuilles['decile'] == 9).sum()

    return portefeuilles, n_losers, n_winners


# =============================================================================
# ÉTAPE 3 — CALCUL DE LA RENTABILITÉ SUR LES 6 MOIS SUIVANTS
# =============================================================================

def calculer_rentabilite_placement(df, portefeuilles, semester_placement):
    """
    On observe la performance des WINNERS et LOSERS sur la période de PLACEMENT
    (1 semestre = 6 mois, immédiatement après la période de formation).

    Étapes :
    a) Filtrer les données sur le semestre de placement
    b) Joindre (merge) avec le DataFrame des portefeuilles pour savoir
       à quel décile appartient chaque titre
    c) Calculer la rentabilité mensuelle MOYENNE de chaque portefeuille
    d) Renvoyer le différentiel Winners - Losers
    """

    # Données sur le semestre de placement
    df_placement = df[df['Semester'] == semester_placement]

    # ── Jointure (merge) ──────────────────────────────────────────────────────
    # On associe à chaque ligne de df_placement le décile (0 ou 9) du titre
    # how='inner' : on ne garde que les titres présents dans les deux tableaux
    df_joint = pd.merge(df_placement, portefeuilles, on='PERMNO', how='inner')

    if df_joint.empty:
        return None

    # ── Rentabilité moyenne par portefeuille ─────────────────────────────────
    # groupby('decile') : groupe par Losers (0) et Winners (9)
    # ['RET'].mean()    : rentabilité mensuelle moyenne sur les 6 mois
    rent_par_portefeuille = df_joint.groupby('decile')['RET'].mean()

    if 0 not in rent_par_portefeuille.index or 9 not in rent_par_portefeuille.index:
        return None

    rent_losers  = rent_par_portefeuille[0]
    rent_winners = rent_par_portefeuille[9]

    # Différentiel = rentabilité de la stratégie long/short
    differentiel = rent_winners - rent_losers

    return differentiel


# =============================================================================
# ÉTAPE 4 — ITÉRATION SUR TOUTES LES PÉRIODES
# =============================================================================

def iterer_sur_toutes_les_periodes(df):
    """
    On répète les étapes 2 et 3 en décalant d'un semestre à chaque fois.

    Schéma d'une itération :
    ┌─────────────────────────────────┬──────────────────┐
    │  FORMATION (12 mois = 2 sem.)   │  PLACEMENT (6m)  │
    │  Semester S  +  Semester S+1    │  Semester S+2    │
    └─────────────────────────────────┴──────────────────┘

    Puis on décale : S+1, S+2 → placement S+3, etc.

    Résultat : liste retMom des différentiels Winners - Losers
    """
    print("=" * 60)
    print("ÉTAPES 2, 3 & 4 — Itération sur toutes les périodes")
    print("=" * 60)

    retMom = []   # liste vide pour stocker les résultats

    semestres = sorted(df['Semester'].unique())
    n_total = len(semestres)

    # On a besoin de 3 semestres consécutifs minimum :
    #   S, S+1 → formation  |  S+2 → placement
    for i, s in enumerate(semestres):
        s_formation_debut = s
        s_formation_fin   = s + 1
        s_placement       = s + 2

        # Vérification que les 3 semestres existent dans les données
        if s_formation_debut not in semestres:
            continue
        if s_formation_fin not in semestres:
            continue
        if s_placement not in semestres:
            continue

        # Étape 2 : construction des portefeuilles
        portefeuilles, n_losers, n_winners = constituer_portefeuilles(
            df, s_formation_debut
        )

        # Étape 3 : calcul de la rentabilité sur le placement
        diff = calculer_rentabilite_placement(df, portefeuilles, s_placement)

        if diff is not None:
            retMom.append(diff)   # ajout à la liste
            print(
                f"  Sem. {s_formation_debut}-{s_formation_fin} → placement {s_placement} | "
                f"Winners={n_winners:3d}  Losers={n_losers:3d} | "
                f"Diff = {diff*100:+.2f}%"
            )

    print(f"\n  Total de périodes analysées : {len(retMom)}")
    print()

    return retMom


# =============================================================================
# ÉTAPE 5 — TEST DE SIGNIFICATIVITÉ (TEST DE STUDENT)
# =============================================================================

def analyser_resultats(retMom):
    """
    On convertit la liste en DataFrame et on calcule :

    1. La rentabilité moyenne de la stratégie
    2. Le test de Student (t-test) pour vérifier si cette moyenne est
       significativement différente de zéro

    ── Qu'est-ce que le test de Student ? ────────────────────────────────────
    On se pose la question : "Est-ce que la rentabilité moyenne obtenue
    pourrait être due au hasard ?"

    - H0 (hypothèse nulle)      : la rentabilité moyenne = 0 (pas d'effet)
    - H1 (hypothèse alternative): la rentabilité moyenne ≠ 0 (effet réel)

    Si p-value < 0.05 → on rejette H0 → la stratégie est significative à 5%
    Si p-value > 0.05 → on ne peut pas rejeter H0 → pas de preuve suffisante

    La t-statistique mesure "combien d'écarts-types la moyenne est éloignée de 0".
    Un |t| > 1.96 correspond approximativement à p < 0.05.
    """
    print("=" * 60)
    print("ÉTAPE 5 — Test de significativité")
    print("=" * 60)

    # Conversion de la liste en DataFrame
    retMom = pd.DataFrame(retMom)
    retMom.columns = ['RETStrat']

    # Statistiques descriptives
    rentabilite_moyenne = retMom['RETStrat'].mean()
    ecart_type          = retMom['RETStrat'].std()
    n_periodes          = len(retMom)

    # Test de Student contre la valeur 0
    # ttest_1samp teste si la moyenne de l'échantillon est différente de `popmean`
    t_stat, p_value = stats.ttest_1samp(retMom['RETStrat'], popmean=0)

    print(f"Nombre de périodes      : {n_periodes}")
    print(f"Rentabilité moyenne     : {rentabilite_moyenne * 100:.4f}% par mois")
    print(f"Rentabilité annualisée  : {rentabilite_moyenne * 12 * 100:.2f}% par an")
    print(f"Écart-type              : {ecart_type * 100:.4f}%")
    print(f"t-statistique           : {t_stat:.4f}")
    print(f"p-value                 : {p_value:.4f}")
    print()

    if p_value < 0.01:
        print("→ Résultat significatif au seuil de 1% (très forte significativité)")
    elif p_value < 0.05:
        print("→ Résultat significatif au seuil de 5%")
    elif p_value < 0.10:
        print("→ Résultat significatif au seuil de 10% seulement")
    else:
        print("→ Résultat NON significatif (p-value > 0.10)")

    print()
    print("Distribution des rentabilités :")
    print(retMom['RETStrat'].describe().apply(lambda x: f"{x*100:.4f}%"))

    return retMom


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # 0. Chargement
    df = charger_donnees(DATA_PATH)

    # 1. Nettoyage
    df = nettoyer_donnees(df)

    # 2-3-4. Itération sur toutes les périodes
    retMom = iterer_sur_toutes_les_periodes(df)

    # 5. Analyse et test de significativité
    resultats = analyser_resultats(retMom)

    print("=" * 60)
    print("Terminé.")
