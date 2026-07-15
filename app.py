# ==========================================================
# app.py
# PART 1
# IMPORTS + MODEL LOADING
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from rdkit import Chem

from torch_geometric.utils.smiles import from_smiles
from torch_geometric.data import Batch

from model import Model
from feature_generator import build_features

import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


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
Predict **Solvation Free Energy (ΔG)** using a
Hybrid Graph Neural Network (GAT + GraphSAGE)
combined with RDKit molecular descriptors.
"""
)

st.markdown("---")


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

st.sidebar.success(f"Running on: {device}")


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_everything():

    config = joblib.load(

        "model_config.pkl"

    )

    feature_order = joblib.load(

        "feature_order.pkl"

    )

    scaler = joblib.load(

        "scaler.pkl"

    )

    model = Model(

        desc_dim=config["input_dim"],

        hidden_dim=config["hidden_dim"],

        heads=config["heads"],

        dropout=config["dropout"]

    )

    checkpoint = torch.load(

        "best_model.pt",

        map_location=device,

        weights_only=False

    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(

                checkpoint["model_state_dict"]

            )

        else:

            model.load_state_dict(

                checkpoint

            )

    else:

        model.load_state_dict(

            checkpoint

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
# GRAPH FUNCTIONS
# ==========================================================

def smiles_to_graph(smiles):

    graph = from_smiles(

        smiles

    )

    graph.x = graph.x.float()

    return graph


def create_batch(

    solute_smiles,

    solvent_smiles

):

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
# PART 2
# FEATURE PREPROCESSING + PREDICTION
# ==========================================================

# ==========================================================
# PREPARE FEATURES
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
    # Convert to DataFrame
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
    # Replace NaN & Inf
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
        # Descriptor Features
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

        # ==========================================================
        # RUN PREDICTION
        # ==========================================================
        
        predictions = []
        
        progress = st.progress(0)
        
        total = len(df)
        
        # Convert selected columns to string
        solute_list = (
        
            df[solute_col]
        
            .fillna("")
        
            .astype(str)
        
            .tolist()
        
        )
        
        solvent_list = (
        
            df[solvent_col]
        
            .fillna("")
        
            .astype(str)
        
            .tolist()
        
        )
        
        for i, (solute, solvent) in enumerate(
        
            zip(solute_list, solvent_list)
        
        ):
        
            try:
        
                pred = predict_deltaG(
        
                    solute.strip(),
        
                    solvent.strip()
        
                )
        
            except Exception:
        
                pred = np.nan
        
            predictions.append(pred)
        
            progress.progress(
        
                (i + 1) / total
        
            )
        
        result_df = df.copy()
        
        result_df["Predicted_DeltaG"] = predictions

# ==========================================================
# CSV VALIDATION
# ==========================================================

def validate_csv(df):

    required = [

        "mol_solute",

        "mol_solvent"

    ]

    missing = [

        col

        for col in required

        if col not in df.columns

    ]

    if len(missing) > 0:

        return False, missing

    return True, []


# ==========================================================
# BATCH PREDICTION
# ==========================================================

def batch_predict(df):

    predictions = []

    progress = st.progress(0)

    total = len(df)

    for i, row in enumerate(df.itertuples()):

        pred = predict_deltaG(

            str(row.mol_solute),

            str(row.mol_solvent)

        )

        predictions.append(pred)

        progress.progress(

            (i + 1) / total

        )

    df["Predicted_DeltaG"] = predictions

    return df

# ==========================================================
# PART 3
# STREAMLIT UI
# ==========================================================

st.markdown("---")

tab1, tab2 = st.tabs(

    [

        "🧪 Single Prediction",

        "📂 Batch Prediction"

    ]

)

# ==========================================================
# TAB 1
# ==========================================================

with tab1:

    st.subheader("Single Prediction")

    col1, col2 = st.columns(2)

    with col1:

        solute = st.text_input(

            "Solute SMILES",

            placeholder="Example: CCO"

        )

    with col2:

        solvent = st.text_input(

            "Solvent SMILES",

            placeholder="Example: O"

        )

    st.write("")

    predict_btn = st.button(

        "Predict ΔG",

        use_container_width=True

    )

    if predict_btn:

        if solute == "" or solvent == "":

            st.warning(

                "Please enter both SMILES."

            )

        elif not valid_smiles(solute):

            st.error(

                "Invalid Solute SMILES"

            )

        elif not valid_smiles(solvent):

            st.error(

                "Invalid Solvent SMILES"

            )

        else:

            with st.spinner(

                "Running Hybrid GNN..."

            ):

                prediction = predict_deltaG(

                    solute,

                    solvent

                )

            if prediction is not None:

                st.success(

                    "Prediction Completed"

                )

                st.metric(

                    "Predicted ΔG",

                    f"{prediction:.4f}"

                )

# ==========================================================
# TAB 2
# ==========================================================

with tab2:

    st.subheader("Batch Prediction")

    uploaded = st.file_uploader(

        "Upload CSV File",

        type=["csv"]

    )

    if uploaded is not None:

        df = pd.read_csv(uploaded)

        st.success(f"Dataset Loaded ({len(df)} rows)")

        st.dataframe(df.head())

        st.markdown("---")

        # ======================================================
        # COLUMN SELECTION
        # ======================================================

        columns = list(df.columns)

        solute_col = st.selectbox(

            "Select Solute SMILES Column",

            columns

        )

        solvent_col = st.selectbox(

            "Select Solvent SMILES Column",

            columns

        )

        actual_col = st.selectbox(

            "Actual ΔG Column (Optional)",

            ["None"] + columns

        )

        st.markdown("---")

        if st.button(

            "Run Batch Prediction",

            use_container_width=True

        ):

            predictions = []

            progress = st.progress(0)

            total = len(df)

            for i, row in enumerate(df.itertuples(index=False)):

                solute = str(

                    getattr(row, solute_col)

                )

                solvent = str(

                    getattr(row, solvent_col)

                )

                pred = predict_deltaG(

                    solute,

                    solvent

                )

                predictions.append(pred)

                progress.progress(

                    (i + 1) / total

                )

            result_df = df.copy()

            result_df["Predicted_DeltaG"] = predictions

            st.success("Prediction Completed")

            st.dataframe(result_df)

            csv = result_df.to_csv(

                index=False

            ).encode("utf-8")

            st.download_button(

                "Download Predictions",

                csv,

                "Predictions.csv",

                "text/csv"

            )

            # ==================================================
            # OPTIONAL TRUE VS PREDICTED
            # ==================================================

            if actual_col != "None":

    actual = pd.to_numeric(

        result_df[actual_col],

        errors="coerce"

    )

    predicted = pd.to_numeric(

        result_df["Predicted_DeltaG"],

        errors="coerce"

    )

    valid = actual.notna() & predicted.notna()

    actual = actual[valid]

    predicted = predicted[valid]

    if len(actual) > 1:

        rmse = np.sqrt(

            mean_squared_error(

                actual,

                predicted

            )

        )

        mae = mean_absolute_error(

            actual,

            predicted

        )

        r2 = r2_score(

            actual,

            predicted

        )

        st.success("Evaluation Completed")

        c1, c2, c3 = st.columns(3)

        c1.metric("RMSE", f"{rmse:.4f}")

        c2.metric("MAE", f"{mae:.4f}")

        c3.metric("R²", f"{r2:.4f}")

        fig, ax = plt.subplots(figsize=(6,6))

        ax.scatter(

            actual,

            predicted,

            alpha=0.6

        )

        mn = min(

            actual.min(),

            predicted.min()

        )

        mx = max(

            actual.max(),

            predicted.max()

        )

        ax.plot(

            [mn, mx],

            [mn, mx],

            "r--",

            linewidth=2

        )

        ax.set_xlabel("Actual ΔG")

        ax.set_ylabel("Predicted ΔG")

        ax.set_title("True vs Predicted")

        ax.grid(True)

        st.pyplot(fig)

    else:

        st.warning(

            "Not enough valid values to calculate metrics."

        )
# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.header(

    "Model"

)

st.sidebar.write(

    "Hybrid GNN"

)

st.sidebar.write(

    "GAT + GraphSAGE"

)

st.sidebar.write(

    f"Descriptors: {len(feature_order)}"

)

st.sidebar.write(

    f"Device: {device}"

)

st.sidebar.markdown("---")

st.sidebar.success(

    "Ready for Prediction"

)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(

    "Hybrid GNN Solvation Free Energy Prediction"
)
