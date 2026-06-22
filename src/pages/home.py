import streamlit as st
from src.utils.design import thick_divider
from src.utils.style import load_css

load_css()

st.set_page_config(
    page_title="ChemFlow",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <div class="home-main-title">
        🧪 ChemFlow
    </div>
    """,
    unsafe_allow_html=True,
)


st.image("assets/images/chemflow_overview.png", use_container_width=True,)


st.divider()


st.markdown("#### 📥 Public Data Sources")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.link_button("UniProt", "https://www.uniprot.org")

with c2:
    st.link_button("ChEMBL", "https://www.ebi.ac.uk/chembl/")

with c3:
    st.link_button("PubChem", "https://pubchem.ncbi.nlm.nih.gov/")

with c4:
    st.link_button("BindingDB", "https://www.bindingdb.org/")

with c5:
    st.link_button("RCSB PDB", "https://www.rcsb.org/")



st.markdown(
        """
        <div style="background:#f8fafc; padding:24px; border-radius:16px; border:1px solid #e5e7eb; box-shadow:0 4px 14px rgba(0,0,0,0.06);">
        <h3 style="color:#0f172a; font-size:26px; font-weight:800;">
        Welcome to ChemFlow: Your Cheminformatics Workflow Platform
        </h3>

        <p style="font-size:16px; line-height:1.6; color:#334155;">
        <strong>ChemFlow</strong> is a portfolio platform for building reproducible cheminformatics workflows, including:
        </p>

        <ul style="font-size:16px; line-height:1.8; color:#334155;">
        <li>Public data extraction from different databases: <strong>ChEMBL</strong>, <strong>BindingDB</strong>, <strong>PubChem</strong>, and <strong>UniProt</strong></li>
        <li>Bioactivity data cleaning, harmonization, and organization</li>
        <li>Molecular visualization and exploratory analysis</li>
        <li>Similarity search and clustering</li>
        <li>Machine learning and deep learning models</li>
        <li>Generative AI for molecular design</li>
        <li>Decision-support dashboards for drug discovery</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()

st.subheader("Platform Modules")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="card">
        <h3>📥 Data Extraction</h3>
        <p>Retrieve public molecular and bioactivity data from ChEMBL, BindingDB, PubChem, and UniProt.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card">
        <h3>🧹 Data Curation</h3>
        <p>Standardize SMILES, remove duplicates, harmonize IC50/Ki/Kd values, and convert activity to pIC50.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="card">
        <h3>📊 Visualization</h3>
        <p>Explore chemical space, activity distributions, subtype selectivity, and assay coverage.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

c4, c5, c6 = st.columns(3)

with c4:
    st.markdown(
        """
        <div class="card">
        <h3>🔎 Similarity Search</h3>
        <p>Use molecular fingerprints, Tanimoto similarity, clustering, and scaffold analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        """
        <div class="card">
        <h3>🤖 ML / DL Modeling</h3>
        <p>Build QSAR, classification, regression, multi-task models, and deep learning predictors.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c6:
    st.markdown(
        """
        <div class="card">
        <h3>🧬 Generative AI</h3>
        <p>Generate, filter, and prioritize new molecules using AI-guided molecular design workflows.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.subheader("Workflow Overview")

st.markdown(
    """
```text
Public Data
    ↓
Data Cleaning & Standardization
    ↓
Bioactivity Harmonization
    ↓
Molecular Representation
    ↓
Visualization & Similarity Search
    ↓
Machine Learning / Deep Learning
    ↓
Generative Molecular Design
    ↓
Decision Support
""")