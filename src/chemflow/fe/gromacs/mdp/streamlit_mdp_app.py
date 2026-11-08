"""
streamlit_mdp_app.py

Interactive test UI for mdp_tools.py. Lets you:
  1. Upload or paste a .mdp file and inspect/edit parameters
  2. Generate a full set of per-lambda-window .mdp files
  3. Run QC checks (lambda-vector consistency, leftover placeholders)
  4. Download the results as a zip

Run with:
    streamlit run streamlit_mdp_app.py

Requires mdp_tools.py to be in the same directory.
"""

import io
import os
import zipfile
import tempfile

import streamlit as st

from mdp_tools import (
    MDPFile,
    generate_lambda_mdps,
    verify_lambda_vectors_consistent,
    check_no_placeholders,
)

st.set_page_config(page_title="MDP Tools Tester", layout="wide")
st.title("GROMACS .mdp Parser / Lambda-Window Generator")

st.markdown(
    "Upload an `.mdp` file (e.g. a FEP template with `init-lambda-state = "
    "{LAMBDA_STATE}`), inspect its parameters, then generate a full set of "
    "per-window files."
)

# ----------------------------------------------------------------------
# Session state setup
# ----------------------------------------------------------------------
if "mdp_text" not in st.session_state:
    st.session_state.mdp_text = ""
if "generated_files" not in st.session_state:
    st.session_state.generated_files = {}  # {relative_path: content}

# ----------------------------------------------------------------------
# Step 1 — Input: upload or paste
# ----------------------------------------------------------------------
st.header("1. Load an .mdp file")

input_mode = st.radio("Input method", ["Upload file", "Paste text", "Use example template"],
                       horizontal=True)

if input_mode == "Upload file":
    uploaded = st.file_uploader("Choose a .mdp file", type=["mdp", "txt"])
    if uploaded is not None:
        st.session_state.mdp_text = uploaded.read().decode("utf-8")

elif input_mode == "Paste text":
    pasted = st.text_area("Paste .mdp content here", height=250,
                           value=st.session_state.mdp_text)
    st.session_state.mdp_text = pasted

else:  # Use example template
    example = """; npt_FEP_AMBER.mdp (example)
define                    = -DPOSRES -DPOSRES_LIG -DPOSRES_COFACTOR

integrator                = md
nsteps                    = 250000
dt                        = 0.002
continuation              = yes

cutoff-scheme             = Verlet
rlist                     = 1.0
coulombtype               = PME
rcoulomb                  = 1.0
vdwtype                   = Cut-off
vdw-modifier              = Potential-shift
rvdw                      = 1.0

constraints               = h-bonds
constraint_algorithm      = LINCS

tcoupl                    = V-rescale
tc-grps                   = SOLU_MEMB SOLV
tau_t                     = 1.0   1.0
ref_t                     = 300   300

gen-vel                   = no

pcoupl                    = berendsen
pcoupltype                = semiisotropic
tau_p                     = 5.0
ref_p                     = 1.0   1.0
compressibility           = 4.5e-5  4.5e-5

; Free energy block
free-energy               = yes
init-lambda-state         = {LAMBDA_STATE}
delta-lambda              = 0
calc-lambda-neighbors     = -1

coul-lambdas              = 0.00 0.00 0.00 0.00 0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 1.00
vdw-lambdas               = 0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
bonded-lambdas            = 0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
restraint-lambdas         = 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00

sc-alpha                  = 0.5
sc-power                  = 1
sc-sigma                  = 0.3
sc-coul                   = no
sc-r-power                = 6

couple-moltype            =
couple-lambda0            = vdw-q
couple-lambda1            = vdw-q
couple-intramol           = no
"""
    st.session_state.mdp_text = example
    st.code(example, language="ini")

if not st.session_state.mdp_text.strip():
    st.info("Load a file above to continue.")
    st.stop()

# ----------------------------------------------------------------------
# Parse it
# ----------------------------------------------------------------------
tmp_dir = tempfile.mkdtemp()
tmp_input_path = os.path.join(tmp_dir, "input.mdp")
with open(tmp_input_path, "w") as f:
    f.write(st.session_state.mdp_text)

try:
    mdp = MDPFile.read(tmp_input_path)
except Exception as e:
    st.error(f"Failed to parse .mdp file: {e}")
    st.stop()

params = mdp.as_dict()

st.success(f"Parsed successfully — {len(params)} parameters found.")

# ----------------------------------------------------------------------
# Step 2 — Inspect / edit parameters
# ----------------------------------------------------------------------
st.header("2. Inspect and edit parameters")

col1, col2 = st.columns([2, 1])

with col1:
    search = st.text_input("Filter parameters (substring match)", "")
    filtered = {k: v for k, v in params.items() if search.lower() in k.lower()}
    st.dataframe(
        {"parameter": list(filtered.keys()), "value": list(filtered.values())},
        use_container_width=True,
        height=300,
    )

with col2:
    st.subheader("Edit a parameter")
    edit_key = st.selectbox("Parameter", options=sorted(params.keys()))
    edit_value = st.text_input("New value", value=params.get(edit_key, ""))
    if st.button("Apply edit"):
        mdp.set(edit_key, edit_value)
        st.session_state.mdp_text = mdp.render()
        st.rerun()

st.download_button(
    "Download edited .mdp",
    data=mdp.render(),
    file_name="edited.mdp",
    mime="text/plain",
)

# ----------------------------------------------------------------------
# Step 3 — Lambda window generation
# ----------------------------------------------------------------------
st.header("3. Generate per-lambda-window files")

has_placeholder = "{LAMBDA_STATE}" in st.session_state.mdp_text
lambda_param_default = "init-lambda-state"

colA, colB, colC = st.columns(3)
with colA:
    n_windows = st.number_input("Number of lambda windows", min_value=1, max_value=64, value=13)
with colB:
    lambda_param = st.text_input("Lambda parameter name", value=lambda_param_default)
with colC:
    output_filename = st.text_input("Output filename (per window)", value="npt.mdp")

if not has_placeholder:
    st.warning(
        f"No `{{LAMBDA_STATE}}` placeholder found in the loaded file. "
        f"Generation will instead directly SET `{lambda_param}` to each window "
        f"index (0..N-1), overwriting whatever value is currently there."
    )

if st.button("Generate lambda windows", type="primary"):
    output_root = os.path.join(tmp_dir, "rbfe_run")
    os.makedirs(output_root, exist_ok=True)

    written_paths = generate_lambda_mdps(
        template_path=tmp_input_path,
        output_dir=output_root,
        n_windows=n_windows,
        output_filename=output_filename,
        lambda_param=lambda_param,
    )

    # Stash generated content in session state for display/download/QC
    st.session_state.generated_files = {}
    for p in written_paths:
        rel = os.path.relpath(p, output_root)
        with open(p) as f:
            st.session_state.generated_files[rel] = f.read()

    st.success(f"Generated {len(written_paths)} files across {n_windows} windows.")

# ----------------------------------------------------------------------
# Step 4 — Show generated files + QC
# ----------------------------------------------------------------------
if st.session_state.generated_files:
    st.header("4. Generated files and QC checks")

    tab1, tab2, tab3 = st.tabs(["Browse files", "QC: placeholders", "QC: lambda-vector consistency"])

    with tab1:
        chosen = st.selectbox("View file", options=sorted(st.session_state.generated_files.keys()))
        st.code(st.session_state.generated_files[chosen], language="ini")

    with tab2:
        # Re-run check_no_placeholders against the in-memory generated content
        import re
        pattern = re.compile(r"\{.*?\}")
        problems = []
        for rel, content in st.session_state.generated_files.items():
            matches = pattern.findall(content)
            if matches:
                problems.append((rel, matches))

        if problems:
            st.error(f"Unreplaced placeholders found in {len(problems)} file(s):")
            for rel, matches in problems:
                st.write(f"- `{rel}`: {matches}")
        else:
            st.success(f"No unreplaced placeholders in {len(st.session_state.generated_files)} files.")

    with tab3:
        vector_keys = [
            "coul-lambdas", "vdw-lambdas", "bonded-lambdas",
            "restraint-lambdas", "sc-alpha", "sc-power", "sc-sigma",
        ]
        # Parse each generated file's text directly (re-use MDPFile by writing to temp)
        results = {}
        parsed_mdps = {}
        for rel, content in st.session_state.generated_files.items():
            tmp_path = os.path.join(tmp_dir, "qc_" + rel.replace(os.sep, "_"))
            with open(tmp_path, "w") as f:
                f.write(content)
            parsed_mdps[rel] = MDPFile.read(tmp_path)

        all_consistent = True
        for key in vector_keys:
            values = {rel: m.get(key) for rel, m in parsed_mdps.items()}
            unique_values = set(values.values())
            consistent = len(unique_values) <= 1
            results[key] = consistent
            if not consistent:
                all_consistent = False
                st.error(f"`{key}` is INCONSISTENT across generated files:")
                st.json(values)

        if all_consistent:
            st.success(f"All {len(vector_keys)} lambda-vector parameters are "
                        f"consistent across {len(parsed_mdps)} files.")

        # Also verify init-lambda-state / chosen lambda_param actually increments correctly
        st.subheader(f"`{lambda_param}` per window")
        lam_values = {}
        for rel, m in parsed_mdps.items():
            lam_values[rel] = m.get(lambda_param)
        st.json(lam_values)

    # ------------------------------------------------------------------
    # Step 5 — Download everything as a zip
    # ------------------------------------------------------------------
    st.header("5. Download")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, content in st.session_state.generated_files.items():
            zf.writestr(rel, content)
    zip_buffer.seek(0)

    st.download_button(
        "Download all generated .mdp files (.zip)",
        data=zip_buffer,
        file_name="lambda_windows.zip",
        mime="application/zip",
    )