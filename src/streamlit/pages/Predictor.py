import streamlit as st
from pathlib import Path
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw

from src.utils.style import load_css as inject_css
from src.streamlit.utils.select_file import file_picker

from src.chemflow.machine_learning.predict.predictor import ChemFlowPredictor


# ============================================================
# Session state
# ============================================================

def init_predictor_state():
    defaults = {
        "predictor": None,
        "predictor_model_file": None,
        "predictor_input_df": None,
        "predictor_results_df": None,
        "predictor_model_info": None,
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
        st.info("Please select a trained ChemFlow `model_package.pkl` file.")
        return None

    model_file = Path(model_file)

    if st.session_state.get("predictor_model_file") == str(model_file):
        predictor = st.session_state.get("predictor")

        if predictor is not None:
            st.success("Model already loaded.")
            show_model_info(st.session_state.get("predictor_model_info"))

        return predictor

    try:
        predictor = ChemFlowPredictor(model_file)
        model_info = predictor.get_model_info()

        st.session_state["predictor"] = predictor
        st.session_state["predictor_model_file"] = str(model_file)
        st.session_state["predictor_model_info"] = model_info

        st.success("Model loaded successfully.")
        show_model_info(model_info)

        return predictor

    except Exception as e:
        st.error(f"Failed to load model:\n{e}")

        st.session_state["predictor"] = None
        st.session_state["predictor_model_file"] = None
        st.session_state["predictor_model_info"] = None

        return None


def show_model_info(model_info):
    if not model_info:
        st.warning("No model metadata found.")
        return

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Model", str(model_info.get("model_name")))

    with c2:
        st.metric("Task", str(model_info.get("task_type")))

    with c3:
        n_bits = model_info.get("n_bits")
        st.metric("FP Bits", "NA" if n_bits is None else str(n_bits))

    st.markdown("#### Feature Information")

    feature_types = model_info.get("feature_types")
    desc_names = model_info.get("desc_names")
    feature_shapes = model_info.get("feature_array_shapes")

    st.write("Feature types:")
    st.code(str(feature_types), language=None)

    if desc_names:
        with st.expander("Descriptor Names", expanded=False):
            st.dataframe(
                pd.DataFrame({"descriptor": desc_names}),
                use_container_width=True,
                hide_index=True,
            )

    if feature_shapes:
        with st.expander("Feature Array Shapes", expanded=False):
            st.json(feature_shapes)

    with st.expander("Full Model Info", expanded=False):
        st.json(model_info)


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

    if max_available <= 0:
        st.info("No molecules to display.")
        return

    if max_available == 1:
        max_mols = 1
    else:
        max_mols = st.slider(
            "Number of molecules to show",
            min_value=1,
            max_value=max_available,
            value=min(max_mols, max_available),
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

            st.dataframe(df, use_container_width=True, hide_index=True)

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
            st.dataframe(df.head(), use_container_width=True)

            smiles_col = st.selectbox(
                "SMILES column for structure display",
                df.columns,
                index=list(df.columns).index("SMILES") if "SMILES" in df.columns else 0,
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

def prediction_panel(predictor):
    st.subheader("Prediction Settings")

    input_df = st.session_state.get("predictor_input_df")

    if predictor is None:
        st.warning("Please load a model first.")
        return

    if input_df is None or input_df.empty:
        st.warning("Please provide input molecules first.")
        return

    default_smiles_col = predictor.feature_config.get("smiles_col", "SMILES")

    smiles_col = st.selectbox(
        "SMILES Column",
        input_df.columns,
        index=list(input_df.columns).index(default_smiles_col)
        if default_smiles_col in input_df.columns
        else list(input_df.columns).index("SMILES")
        if "SMILES" in input_df.columns
        else 0,
        key="predictor_smiles_col",
    )

    st.markdown("#### Model Feature Types")
    st.code(str(predictor.feature_types), language=None)

    st.markdown("#### Input Preview")
    st.dataframe(input_df.head(20), use_container_width=True)

    show_molecule_structures(
        df=input_df,
        smiles_col=smiles_col,
        max_mols=8,
        key_prefix="predictor_input_preview",
    )

    if st.button(
        "Run Prediction",
        type="primary",
        use_container_width=True,
        key="predictor_run_button",
    ):
        try:
            results_df = predictor.predict_from_dataframe(
                input_df,
                smiles_col=smiles_col,
            )

            st.session_state["predictor_results_df"] = results_df

            st.success("Prediction finished.")
            st.dataframe(results_df, use_container_width=True)

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
        return

    st.dataframe(results_df, use_container_width=True)

    csv_data = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv_data,
        file_name="chemflow_predictions.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_prediction_results",
    )

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
            key_prefix="predictor_results",
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

    predictor = load_model_panel()

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
        prediction_panel(predictor)

    with tab3:
        results_panel()


if __name__ == "__main__":
    design()