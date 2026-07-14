# =============================================================================
# feature_generator.py
# PART 1
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    GraphDescriptors,
    Crippen,
    rdMolDescriptors,
    Descriptors3D
)

# =============================================================================
# SAFE DIVISION
# =============================================================================

def safe_div(a, b):

    try:

        if pd.isna(a):

            return np.nan

        if pd.isna(b):

            return np.nan

        if abs(float(b)) < 1e-12:

            return 0.0

        return float(a) / float(b)

    except:

        return np.nan


# =============================================================================
# TOPOLOGICAL INDICES
# =============================================================================

def wiener_index(mol):

    dmat = Chem.GetDistanceMatrix(mol)

    total = 0.0

    n = dmat.shape[0]

    for i in range(n):

        for j in range(i + 1, n):

            total += dmat[i, j]

    return float(total)


def zagreb_index(mol):

    value = 0.0

    for atom in mol.GetAtoms():

        value += atom.GetDegree() ** 2

    return float(value)


def randic_index(mol):

    value = 0.0

    for bond in mol.GetBonds():

        d1 = bond.GetBeginAtom().GetDegree()

        d2 = bond.GetEndAtom().GetDegree()

        if d1 > 0 and d2 > 0:

            value += 1.0 / np.sqrt(d1 * d2)

    return float(value)


def petitjean_index(mol):

    dmat = Chem.GetDistanceMatrix(mol)

    ecc = dmat.max(axis=1)

    diameter = ecc.max()

    radius = ecc.min()

    if diameter == 0:

        return 0.0

    return (diameter - radius) / diameter


# =============================================================================
# SMARTS PATTERNS
# =============================================================================

FG_SMARTS = {

    "Carboxylic acid": "[CX3](=O)[OX2H1]",

    "Amide": "[NX3][CX3](=O)",

    "Ester": "[CX3](=O)[OX2][#6]",

    "Alcohol": "[OX2H][CX4]",

    "Phenol": "c[OX2H]",

    "Primary amine": "[NX3;H2]",

    "Secondary amine": "[NX3;H1]",

    "Tertiary amine": "[NX3;H0]",

    "Ketone": "[#6][CX3](=O)[#6]",

    "Aldehyde": "[CX3H1](=O)",

    "Ether": "[OD2]([#6])[#6]",

    "Nitro": "[NX3](=O)=O",

    "Nitrile": "C#N",

    "Sulfoxide": "S(=O)",

    "Sulfone": "S(=O)(=O)",

    "Thiol": "[SX2H]",

    "Phosphate": "P(=O)(O)(O)",

    "Halide": "[F,Cl,Br,I]"

}


# =============================================================================
# DOMINANT FUNCTIONAL GROUP
# =============================================================================

def detect_dominant_functional_group(mol):

    for name, smarts in FG_SMARTS.items():

        patt = Chem.MolFromSmarts(smarts)

        if patt is not None and mol.HasSubstructMatch(patt):

            return name

    return "Hydrocarbon"


# =============================================================================
# HYDROGEN BOND CLASS
# =============================================================================

def hydrogen_bond_class(mol):

    hbd = rdMolDescriptors.CalcNumHBD(mol)

    hba = rdMolDescriptors.CalcNumHBA(mol)

    if hbd > 0 and hba > 0:

        return "Donor & Acceptor"

    elif hbd > 0:

        return "Donor"

    elif hba > 0:

        return "Acceptor"

    else:

        return "None"


# =============================================================================
# POLARITY CLASS
# =============================================================================

def polarity_class(logp, tpsa):

    if tpsa < 20:

        return "Nonpolar"

    elif tpsa < 40:

        return "Weakly polar"

    elif tpsa < 75:

        return "Moderately polar"

    else:

        return "Highly polar"


# =============================================================================
# MOLECULAR FAMILY
# =============================================================================

def molecular_family(hbd, hba):

    if hbd > 0:

        return "Polar protic"

    elif hba > 0:

        return "Polar aprotic"

    else:

        return "Nonpolar"


# =============================================================================
# GENERATE 3D MOLECULE
# =============================================================================

def generate_3d_mol(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()

    params.randomSeed = 42

    status = AllChem.EmbedMolecule(

        mol,

        params

    )

    if status != 0:

        status = AllChem.EmbedMolecule(

            mol,

            useRandomCoords=True

        )

    if status != 0:

        return None

    try:

        if AllChem.MMFFHasAllMoleculeParams(mol):

            AllChem.MMFFOptimizeMolecule(mol)

        else:

            AllChem.UFFOptimizeMolecule(mol)

    except:

        pass

    return mol


# =============================================================================
# END OF PART 1
# =============================================================================
# =============================================================================
# PART 2A
# SOLVATION DESCRIPTORS (2D + GRAPH DESCRIPTORS)
# =============================================================================

def solvation_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    desc = {}

    # ==========================================================
    # BASIC MOLECULAR PROPERTIES
    # ==========================================================

    logp = Crippen.MolLogP(mol)

    tpsa = rdMolDescriptors.CalcTPSA(mol)

    hbd = rdMolDescriptors.CalcNumHBD(mol)

    hba = rdMolDescriptors.CalcNumHBA(mol)

    # ==========================================================
    # BASIC DESCRIPTORS
    # ==========================================================

    desc["MolWt"] = Descriptors.MolWt(mol)

    desc["ExactMolWt"] = Descriptors.ExactMolWt(mol)

    desc["LogP"] = logp

    desc["TPSA"] = tpsa

    desc["MolMR"] = Crippen.MolMR(mol)

    desc["HeavyAtomCount"] = mol.GetNumHeavyAtoms()

    desc["NumAtoms"] = mol.GetNumAtoms()

    desc["NumHeavyAtoms"] = mol.GetNumHeavyAtoms()

    desc["NumRotatableBonds"] = \
        rdMolDescriptors.CalcNumRotatableBonds(mol)

    desc["NumRings"] = \
        rdMolDescriptors.CalcNumRings(mol)

    desc["NumAromaticRings"] = \
        rdMolDescriptors.CalcNumAromaticRings(mol)

    desc["NumAliphaticRings"] = \
        rdMolDescriptors.CalcNumAliphaticRings(mol)

    desc["NumSaturatedRings"] = \
        rdMolDescriptors.CalcNumSaturatedRings(mol)

    desc["FractionCSP3"] = \
        rdMolDescriptors.CalcFractionCSP3(mol)

    desc["FormalCharge"] = \
        Chem.GetFormalCharge(mol)

    # ==========================================================
    # HYDROGEN BOND
    # ==========================================================

    desc["HBD"] = hbd

    desc["HBA"] = hba

    desc["HydrogenBondClass"] = \
        hydrogen_bond_class(mol)

    desc["DominantFunctionalGroup"] = \
        detect_dominant_functional_group(mol)

    desc["Family"] = \
        molecular_family(hbd, hba)

    desc["PolarityClass"] = \
        polarity_class(logp, tpsa)

    # ==========================================================
    # GRAPH DESCRIPTORS
    # ==========================================================

    graph_functions = {

        "BalabanJ":
            GraphDescriptors.BalabanJ,

        "BertzCT":
            GraphDescriptors.BertzCT,

        "Chi0":
            GraphDescriptors.Chi0,

        "Chi1":
            GraphDescriptors.Chi1,

        "HallKierAlpha":
            GraphDescriptors.HallKierAlpha,

        "Kappa1":
            GraphDescriptors.Kappa1,

        "Kappa2":
            GraphDescriptors.Kappa2,

    }

    for name, func in graph_functions.items():

        try:

            desc[name] = func(mol)

        except:

            desc[name] = np.nan

    # ==========================================================
    # TOPOLOGICAL INDICES
    # ==========================================================

    try:

        desc["WienerIndex"] = \
            wiener_index(mol)

    except:

        desc["WienerIndex"] = np.nan

    try:

        desc["ZagrebIndex"] = \
            zagreb_index(mol)

    except:

        desc["ZagrebIndex"] = np.nan

    try:

        desc["RandicIndex"] = \
            randic_index(mol)

    except:

        desc["RandicIndex"] = np.nan

    try:

        desc["PetitjeanIndex"] = \
            petitjean_index(mol)

    except:

        desc["PetitjeanIndex"] = np.nan

    # ==========================================================
    # GENERATE 3D MOLECULE
    # ==========================================================

    mol3d = generate_3d_mol(smiles)

    if mol3d is None:

        desc["MolVolume"] = np.nan
        desc["LabuteASA"] = np.nan
        desc["RadiusOfGyration"] = np.nan
        desc["Asphericity"] = np.nan
        desc["Eccentricity"] = np.nan
        desc["InertialShapeFactor"] = np.nan
        desc["SpherocityIndex"] = np.nan
        desc["PMI1"] = np.nan
        desc["PMI2"] = np.nan
        desc["PMI3"] = np.nan
        desc["PMI_ratio_1_2"] = np.nan
        desc["PMI_ratio_2_3"] = np.nan

        return desc

    # ==========================================================
    # PART 2B CONTINUES HERE
    # ==========================================================
      # ==========================================================
    # PART 2B
    # 3D DESCRIPTORS
    # ==========================================================

    # ----------------------------------------------------------
    # Molecular Volume
    # ----------------------------------------------------------

    try:

        desc["MolVolume"] = AllChem.ComputeMolVolume(
            mol3d
        )

    except:

        desc["MolVolume"] = np.nan

    # ----------------------------------------------------------
    # Labute Surface Area
    # ----------------------------------------------------------

    try:

        desc["LabuteASA"] = rdMolDescriptors.CalcLabuteASA(
            mol
        )

    except:

        desc["LabuteASA"] = np.nan

    # ----------------------------------------------------------
    # Radius of Gyration
    # ----------------------------------------------------------

    try:

        desc["RadiusOfGyration"] = \
            Descriptors3D.RadiusOfGyration(
                mol3d
            )

    except:

        desc["RadiusOfGyration"] = np.nan

    # ----------------------------------------------------------
    # Asphericity
    # ----------------------------------------------------------

    try:

        desc["Asphericity"] = \
            Descriptors3D.Asphericity(
                mol3d
            )

    except:

        desc["Asphericity"] = np.nan

    # ----------------------------------------------------------
    # Eccentricity
    # ----------------------------------------------------------

    try:

        desc["Eccentricity"] = \
            Descriptors3D.Eccentricity(
                mol3d
            )

    except:

        desc["Eccentricity"] = np.nan

    # ----------------------------------------------------------
    # Inertial Shape Factor
    # ----------------------------------------------------------

    try:

        desc["InertialShapeFactor"] = \
            Descriptors3D.InertialShapeFactor(
                mol3d
            )

    except:

        desc["InertialShapeFactor"] = np.nan

    # ----------------------------------------------------------
    # Spherocity Index
    # ----------------------------------------------------------

    try:

        desc["SpherocityIndex"] = \
            Descriptors3D.SpherocityIndex(
                mol3d
            )

    except:

        desc["SpherocityIndex"] = np.nan

    # ----------------------------------------------------------
    # Principal Moments of Inertia
    # ----------------------------------------------------------

    try:

        desc["PMI1"] = \
            Descriptors3D.PMI1(
                mol3d
            )

    except:

        desc["PMI1"] = np.nan

    try:

        desc["PMI2"] = \
            Descriptors3D.PMI2(
                mol3d
            )

    except:

        desc["PMI2"] = np.nan

    try:

        desc["PMI3"] = \
            Descriptors3D.PMI3(
                mol3d
            )

    except:

        desc["PMI3"] = np.nan

    # ----------------------------------------------------------
    # PMI Ratios
    # ----------------------------------------------------------

    desc["PMI_ratio_1_2"] = safe_div(

        desc["PMI1"],

        desc["PMI2"]

    )

    desc["PMI_ratio_2_3"] = safe_div(

        desc["PMI2"],

        desc["PMI3"]

    )

    # ==========================================================
    # REMOVE INF VALUES
    # ==========================================================

    for key, value in desc.items():

        if isinstance(
            value,
            (float, int, np.floating, np.integer)
        ):

            if np.isinf(value):

                desc[key] = np.nan

    return desc


# ==========================================================
# END OF PART 2
# ==========================================================
# =============================================================================
# PART 3
# BUILD FEATURE VECTOR
# =============================================================================

def build_features(

    solute_smiles,

    solvent_smiles

):

    # ----------------------------------------------------------
    # Generate descriptors
    # ----------------------------------------------------------

    solute = solvation_descriptors(

        solute_smiles

    )

    solvent = solvation_descriptors(

        solvent_smiles

    )

    if solute is None:

        return None

    if solvent is None:

        return None

    features = {}

    # ==========================================================
    # Store SMILES
    # ==========================================================

    features["mol_solute"] = solute_smiles

    features["mol_solvent"] = solvent_smiles

    # ==========================================================
    # Solute Descriptors
    # ==========================================================

    for key, value in solute.items():

        features[f"solute_{key}"] = value

    # ==========================================================
    # Solvent Descriptors
    # ==========================================================

    for key, value in solvent.items():

        features[f"solvent_{key}"] = value

    # ==========================================================
    # Numeric Descriptor Names
    # ==========================================================

    numeric_keys = []

    for key, value in solute.items():

        if isinstance(

            value,

            (

                int,

                float,

                np.integer,

                np.floating

            )

        ):

            numeric_keys.append(key)

    # ==========================================================
    # Interaction Features
    # ==========================================================

    for key in numeric_keys:

        if key not in solvent:

            continue

        try:

            s = float(solute[key])

            v = float(solvent[key])

        except:

            continue

        # ------------------------------------------------------
        # Difference
        # ------------------------------------------------------

        features[f"diff_{key}"] = abs(

            s - v

        )

        # ------------------------------------------------------
        # Product
        # ------------------------------------------------------

        features[f"prod_{key}"] = s * v

        # ------------------------------------------------------
        # Ratio
        # ------------------------------------------------------

        features[f"ratio_{key}"] = safe_div(

            s,

            v

        )

    # ==========================================================
    # Replace Inf
    # ==========================================================

    for key, value in features.items():

        if isinstance(

            value,

            (

                int,

                float,

                np.integer,

                np.floating

            )

        ):

            if np.isinf(value):

                features[key] = np.nan

    # ==========================================================
    # Convert NaN → 0
    # ==========================================================

    for key, value in features.items():

        if isinstance(

            value,

            (

                float,

                np.floating

            )

        ):

            if np.isnan(value):

                features[key] = 0.0

    return features


# =============================================================================
# SINGLE PREDICTION DATAFRAME
# =============================================================================

def feature_dataframe(

    solute_smiles,

    solvent_smiles

):

    feat = build_features(

        solute_smiles,

        solvent_smiles

    )

    if feat is None:

        return None

    return pd.DataFrame(

        [feat]

    )


# =============================================================================
# END OF PART 3
# =============================================================================
# =============================================================================
# PART 4
# BATCH FEATURE GENERATION
# =============================================================================

from tqdm import tqdm


def batch_feature_generation(

    df,

    solute_col="mol_solute",

    solvent_col="mol_solvent"

):

    rows = []

    failed = []

    for idx, row in tqdm(

        df.iterrows(),

        total=len(df)

    ):

        try:

            feat = build_features(

                str(row[solute_col]),

                str(row[solvent_col])

            )

            if feat is None:

                failed.append(idx)

                continue

            rows.append(feat)

        except Exception:

            failed.append(idx)

            continue

    feature_df = pd.DataFrame(

        rows

    )

    return feature_df, failed


# =============================================================================
# LOAD CSV
# =============================================================================

def generate_feature_dataframe(

    csv_file,

    solute_col="mol_solute",

    solvent_col="mol_solvent"

):

    df = pd.read_csv(

        csv_file

    )

    feature_df, failed = batch_feature_generation(

        df,

        solute_col,

        solvent_col

    )

    return (

        feature_df,

        failed

    )


# =============================================================================
# SAVE FEATURES
# =============================================================================

def save_features(

    feature_df,

    output_file="generated_features.csv"

):

    feature_df.to_csv(

        output_file,

        index=False

    )

    return output_file


# =============================================================================
# NULL REPORT
# =============================================================================

def null_report(

    feature_df

):

    report = pd.DataFrame({

        "Column": feature_df.columns,

        "NullCount": feature_df.isna().sum().values

    })

    report = report.sort_values(

        "NullCount",

        ascending=False

    )

    return report


# =============================================================================
# END OF FILE
# =============================================================================
