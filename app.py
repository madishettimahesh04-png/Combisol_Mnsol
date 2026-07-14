# ==========================================================
# app.py
# PART 1
# IMPORTS + MODEL LOADING
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from rdkit import Chem
from rdkit.Chem import Draw

from torch_geometric.utils.smiles import from_smiles
from torch_geometric.data import Batch

from model import Model
from feature_generator import build_features


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(

    page_title="Hybrid GNN ΔG Predictor",

    page_icon="🧪",

    layout="wide"

)


# ==========================================================
# TITLE
# ==========================================================

st.title("🧪 Hybrid GNN Solvation Free Energy Predictor")

st.write(
    """
Predict ΔG using

• Graph Neural Network

• RDKit Molecular Descriptors

• Solute + Solvent SMILES
"""
)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


# ==========================================================
# CACHE MODEL
# ==========================================================

@st.cache_resource
def load_everything():

    # ---------------------------------------------
    # Model configuration
    # ---------------------------------------------

    config = joblib.load(
        "model_config.pkl"
    )

    # ---------------------------------------------
    # Feature order
    # ---------------------------------------------

    feature_order = joblib.load(
        "feature_order.pkl"
    )

    # ---------------------------------------------
    # Scaler
    # ---------------------------------------------

    scaler = joblib.load(
        "scaler.pkl"
    )

    # ---------------------------------------------
    # Build model
    # ---------------------------------------------

    model = Model(

        desc_dim=config["input_dim"],

        hidden_dim=config["hidden_dim"],

        heads=config["heads"],

        dropout=config["dropout"]

    )

    state_dict = torch.load(

        "best_model.pt",

        map_location=device,

        weights_only=True

    )

    model.load_state_dict(

        state_dict

    )

    model.to(device)

    model.eval()

    return (

        model,

        scaler,

        feature_order

    )


model, scaler, feature_order = load_everything()


# ==========================================================
# GRAPH CREATION
# ==========================================================

def smiles_to_graph(smiles):

    graph = from_smiles(smiles)

    graph.x = graph.x.float()

    return graph


# ==========================================================
# BATCH GRAPH
# ==========================================================

def create_batch(solute_smiles, solvent_smiles):

    g1 = smiles_to_graph(

        solute_smiles

    )

    g2 = smiles_to_graph(

        solvent_smiles

    )

    return (

        Batch.from_data_list([g1]),

        Batch.from_data_list([g2])

    )


# ==========================================================
# RDKIT IMAGE
# ==========================================================

def molecule_image(smiles):

    mol = Chem.MolFromSmiles(

        smiles

    )

    if mol is None:

        return None

    return Draw.MolToImage(

        mol,

        size=(300,300)

    )
# ==========================================================
# PART 2
# FEATURE PREPROCESSING + PREDICTION
# ==========================================================

# ==========================================================
# FEATURE PREPROCESSING
# ==========================================================

def prepare_features(
    solute_smiles,
    solvent_smiles
):

    # ---------------------------------------------
    # Generate descriptors
    # ---------------------------------------------

    features = build_features(

        solute_smiles,

        solvent_smiles

    )

    if features is None:

        return None

    # ---------------------------------------------
    # Convert to dataframe
    # ---------------------------------------------

    df = pd.DataFrame(

        [features]

    )

    # ---------------------------------------------
    # Remove SMILES columns
    # ---------------------------------------------

    df = df.drop(

        columns=[

            "mol_solute",

            "mol_solvent"

        ],

        errors="ignore"

    )

    # ---------------------------------------------
    # One-hot encoding
    # ---------------------------------------------

    df = pd.get_dummies(

        df

    )

    # ---------------------------------------------
    # Add missing columns
    # ---------------------------------------------

    for col in feature_order:

        if col not in df.columns:

            df[col] = 0

    # ---------------------------------------------
    # Remove extra columns
    # ---------------------------------------------

    df = df[feature_order]

    # ---------------------------------------------
    # Replace NaN
    # ---------------------------------------------

    df = df.replace(

        [

            np.inf,

            -np.inf

        ],

        np.nan

    )

    df = df.fillna(0)

    # ---------------------------------------------
    # Scale
    # ---------------------------------------------

    X = scaler.transform(

        df

    )

    X = np.asarray(

        X,

        dtype=np.float32

    )

    return X


# ==========================================================
# PREDICTION
# ==========================================================

def predict_deltaG(

    solute_smiles,

    solvent_smiles

):

    try:

        # -----------------------------------------
        # Graphs
        # -----------------------------------------

        g1, g2 = create_batch(

            solute_smiles,

            solvent_smiles

        )

        g1 = g1.to(device)

        g2 = g2.to(device)

        # -----------------------------------------
        # Descriptor vector
        # -----------------------------------------

        X = prepare_features(

            solute_smiles,

            solvent_smiles

        )

        if X is None:

            return None

        X = torch.tensor(

            X,

            dtype=torch.float32,

            device=device

        )

        # -----------------------------------------
        # Prediction
        # -----------------------------------------

        with torch.no_grad():

            with torch.amp.autocast("cuda"):

                pred = model(

                    g1,

                    g2,

                    X

                )

        return float(

            pred.cpu().item()

        )

    except Exception as e:

        st.error(

            str(e)

        )

        return None


# ==========================================================
# SMILES VALIDATION
# ==========================================================

def valid_smiles(smiles):

    try:

        mol = Chem.MolFromSmiles(

            smiles

        )

        return mol is not None

    except:

        return False
# ==========================================================
# PART 3
# STREAMLIT USER INTERFACE
# ==========================================================

st.sidebar.header("Model Information")

st.sidebar.write(f"**Device:** {device}")

st.sidebar.write(f"**Descriptors:** {len(feature_order)}")

st.sidebar.write("**Model:** Hybrid GNN")

st.sidebar.write("**Graph Network:** GAT + GraphSAGE")

st.sidebar.markdown("---")

mode = st.sidebar.radio(

    "Prediction Mode",

    [

        "Single Prediction",

        "Batch Prediction"

    ]

)

# ==========================================================
# SINGLE PREDICTION
# ==========================================================

if mode == "Single Prediction":

    st.header("Single Prediction")

    col1, col2 = st.columns(2)

    with col1:

        solute = st.text_input(

            "Solute SMILES",

            "CCO"

        )

    with col2:

        solvent = st.text_input(

            "Solvent SMILES",

            "O"

        )

    st.markdown("---")

    img1, img2 = st.columns(2)

    with img1:

        st.write("### Solute")

        img = molecule_image(solute)

        if img is not None:

            st.image(img)

    with img2:

        st.write("### Solvent")

        img = molecule_image(solvent)

        if img is not None:

            st.image(img)

    st.markdown("---")

    if st.button("Predict ΔG", use_container_width=True):

        if not valid_smiles(solute):

            st.error("Invalid Solute SMILES")

        elif not valid_smiles(solvent):

            st.error("Invalid Solvent SMILES")

        else:

            with st.spinner("Generating descriptors and predicting..."):

                pred = predict_deltaG(

                    solute,

                    solvent

                )

            if pred is not None:

                st.success("Prediction Complete")

                st.metric(

                    "Predicted ΔG",

                    f"{pred:.4f}"

                )

# ==========================================================
# BATCH PREDICTION
# ==========================================================

else:

    st.header("Batch Prediction")

    uploaded = st.file_uploader(

        "Upload CSV",

        type=["csv"]

    )

    st.info(

        "Required columns:\n"

        "• mol_solute\n"

        "• mol_solvent"

    )

    if uploaded is not None:

        df = pd.read_csv(uploaded)

        st.write(df.head())

        if st.button(

            "Run Batch Prediction",

            use_container_width=True

        ):

            predictions = []

            progress = st.progress(0)

            total = len(df)

            for i, row in enumerate(df.itertuples()):

                solute = str(row.mol_solute)

                solvent = str(row.mol_solvent)

                pred = predict_deltaG(

                    solute,

                    solvent

                )

                predictions.append(pred)

                progress.progress(

                    (i + 1) / total

                )

            df["Predicted_DeltaG"] = predictions

            st.success("Prediction Finished")

            st.dataframe(df)

            csv = df.to_csv(

                index=False

            ).encode("utf-8")

            st.download_button(

                "Download Predictions",

                csv,

                "predictions.csv",

                "text/csv"

            )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(

    "Hybrid GNN for Solvation Free Energy Prediction"

)
