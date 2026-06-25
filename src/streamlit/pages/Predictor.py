import streamlit as st
from pathlib import Path
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw

from src.utils.style import load_css as inject_css
from src.streamlit.utils.select_file import file_picker
from src.chemflow.machine_learning.predictor.load_model import load_pickle_model


# ============================================================
# Session state
# ============================================================

def init_predictor_state():
    defaults = {
        "predictor_model": None,
        "predictor_model_file": None,
        "predictor_input_df": None,
        "predictor_results_df": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# Model loading
# ============================================================

def load_model_panel():
    st.subheader("Load Model")

    model_file = file_picker(
        start_dir=Path.cwd(),
        allowed_extensions=(".pkl", ".pickle"),
        key_prefix="predictor_model",
    )

    if not model_file:
        st.info("Please select a trained `.pkl` model file.")
        return None

    if st.session_state.get("predictor_model_file") == str(model_file):
        model = st.session_state.get("predictor_model")
        if model is not None:
            st.success("Model already loaded.")
            st.write("Model Type:")
            st.code(type(model).__name__, language=None)
        return model

    try:
        model = load_pickle_model(model_file)

        st.session_state["predictor_model"] = model
        st.session_state["predictor_model_file"] = str(model_file)

        st.success("Model loaded successfully.")
        st.write("Model Type:")
        st.code(type(model).__name__, language=None)

        if hasattr(model, "n_features_in_"):
            st.metric("Number of Features", model.n_features_in_)

        if hasattr(model, "feature_names_in_"):
            st.write("Features")
            feature_df = pd.DataFrame(
                {"feature": list(model.feature_names_in_)}
            )
            st.dataframe(feature_df, width="stretch")

        if hasattr(model, "classes_"):
            st.write("Classes")
            st.code(str(model.classes_), language=None)

        return model

    except Exception as e:
        st.error(f"Failed to load model:\n{e}")
        st.session_state["predictor_model"] = None
        st.session_state["predictor_model_file"] = None
        return None


# ============================================================
# Structure display
# ============================================================

def show_molecule_structures(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    max_mols: int = 12,
    key_prefix: str = "structures",
):
    st.subheader("Molecular Structures")

    if df is None or df.empty:
        st.info("No molecules to display.")
        return

    if smiles_col not in df.columns:
        st.warning(f"No SMILES column found: {smiles_col}")
        return

    max_available = min(40, len(df))
    default_value = min(max_mols, max_available)

    max_available = min(40, len(df))

    if max_available <= 1:
        max_mols = 1
    else:
        default_value = min(max_mols, max_available)

        max_mols = st.slider(
            "Number of molecules to show",
            min_value=1,
            max_value=max_available,
            value=default_value,
            step=1,
            key=f"{key_prefix}_{smiles_col}_structure_max_mols",
        )

    show_df = df.head(max_mols).copy()
    cols = st.columns(4)

    for i, (_, row) in enumerate(show_df.iterrows()):
        smiles = row.get(smiles_col, "")
        mol = Chem.MolFromSmiles(str(smiles))

        with cols[i % 4]:
            if mol is None:
                st.warning("Invalid SMILES")
                st.caption(str(smiles))
            else:
                img = Draw.MolToImage(mol, size=(250, 200))
                st.image(img)
                st.caption(str(smiles))


# ============================================================
# Input molecules
# ============================================================

def get_smiles_input():
    input_mode = st.radio(
        "Input method",
        [
            "SMILES text",
            "Upload CSV",
            "Use curated database",
        ],
        horizontal=True,
        key="predictor_input_mode",
    )

    if input_mode == "SMILES text":
        smiles_text = st.text_area(
            "Enter SMILES",
            placeholder="CCO\nCCN\nc1ccccc1",
            height=180,
            key="predictor_smiles_text",
        )

        smiles_list = [
            s.strip()
            for s in smiles_text.splitlines()
            if s.strip()
        ]

        if smiles_list:
            df = pd.DataFrame({"SMILES": smiles_list})
            st.session_state["predictor_input_df"] = df

            st.dataframe(df, width="stretch")
            show_molecule_structures(
                df=df,
                smiles_col="SMILES",
                max_mols=12,
                key_prefix="input_text",
            )

    elif input_mode == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            key="predictor_csv_upload",
        )

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state["predictor_input_df"] = df

            st.success("CSV uploaded.")
            st.dataframe(df.head(), width="stretch")

            smiles_col = st.selectbox(
                "SMILES column for structure display",
                df.columns,
                index=list(df.columns).index("SMILES")
                if "SMILES" in df.columns
                else 0,
                key="predictor_upload_smiles_col",
            )

            show_molecule_structures(
                df=df,
                smiles_col=smiles_col,
                max_mols=12,
                key_prefix="upload_csv",
            )

    else:
        st.info("Use molecules from your curated ChemFlow database.")


# ============================================================
# Prediction
# ============================================================

def prediction_panel(model):
    st.subheader("Prediction Settings")

    input_df = st.session_state.get("predictor_input_df")

    if model is None:
        st.warning("Please load a model first.")
        return

    if input_df is None or input_df.empty:
        st.warning("Please provide input molecules first.")
        return

    smiles_col = st.selectbox(
        "SMILES Column",
        input_df.columns,
        index=list(input_df.columns).index("SMILES")
        if "SMILES" in input_df.columns
        else 0,
        key="predictor_smiles_col",
    )

    st.selectbox(
        "Prediction Output Type",
        [
            "Auto-detect",
            "Regression",
            "Classification",
        ],
        key="predictor_prediction_type",
    )

    st.markdown("#### Input Preview")
    st.dataframe(input_df.head(20), width="stretch")

    show_molecule_structures(
        df=input_df,
        smiles_col=smiles_col,
        max_mols=8,
        key_prefix="predictor_input_preview"
    )

    if st.button(
        "Run Prediction",
        type="primary",
        width="stretch",
        key="predictor_run_button",
    ):
        try:
            from src.chemflow.machine_learning.predictor.featurize import featurize_smiles_2057, featurize_smiles

            smiles_list = input_df[smiles_col].astype(str).tolist()

            #X = featurize_smiles_2057(smiles_list)
            X = featurize_smiles(smiles_list)

            preds = model.predict(X)

            results_df = input_df.copy()
            results_df["prediction"] = preds

            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)

                    if proba.ndim == 2:
                        for i in range(proba.shape[1]):
                            results_df[f"prob_class_{i}"] = proba[:, i]

                        results_df["confidence"] = proba.max(axis=1)

                except Exception:
                    pass

            st.session_state["predictor_results_df"] = results_df

            st.success("Prediction finished.")
            st.dataframe(results_df, width="stretch")

        except Exception as e:
            st.error(f"Prediction failed:\n{e}")


# ============================================================
# Results
# ============================================================

def results_panel():
    st.subheader("Prediction Results")

    results_df = st.session_state.get("predictor_results_df")

    if results_df is None or results_df.empty:
        st.info("Prediction results will appear here.")
        st.markdown(
            """
            Expected output:

            - SMILES
            - Predicted value
            - Prediction probability
            - Confidence score
            - Applicability domain flag
            - Molecular properties
            """
        )
        return

    st.dataframe(results_df, width="stretch")

    smiles_candidates = [
        col
        for col in results_df.columns
        if "smiles" in col.lower()
    ]

    if smiles_candidates:
        smiles_col = st.selectbox(
            "SMILES column for result structures",
            results_df.columns,
            index=list(results_df.columns).index(smiles_candidates[0]),
            key="predictor_result_smiles_col",
        )

        show_molecule_structures(
            df=results_df,
            smiles_col=smiles_col,
            max_mols=12,
        )


# ============================================================
# Main page
# ============================================================

def design():
    inject_css()
    init_predictor_state()

    st.markdown(
        """
        <div class="page-title">
            Molecular Property Predictor
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    model = load_model_panel()

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "Input Molecules",
            "Prediction",
            "Results",
        ]
    )

    with tab1:
        st.subheader("Input Molecules")
        get_smiles_input()

    with tab2:
        prediction_panel(model)

    with tab3:
        results_panel()


if __name__ == "__main__":
    design()