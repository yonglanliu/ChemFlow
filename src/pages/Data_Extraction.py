# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.utils.style import load_css as inject_css

import src.tabs.page_data_extraction.tab1 as tab1
import src.tabs.page_data_extraction.tab2 as tab2
import src.tabs.page_data_extraction.tab3 as tab3
import src.tabs.page_data_extraction.tab4 as tab4


# ============================================================
# Page Setup
# ============================================================

inject_css()


st.markdown(
    """
    <div class="page-title">
        Public Database Extraction
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Query, retrieve, and curate public bioactivity and
        structural data from UniProt, ChEMBL, PubChem,
        BindingDB, and RCSB PDB.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# Tabs
# ============================================================

tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs(
    [
        "UniProt",
        "ChEMBL",
        "PubChem",
        "BindingDB",
        "RCSB",
    ]
)

# ============================================================
# UniProt
# ============================================================

with tab_1:
    tab1.design()

# ============================================================
# ChEMBL
# ============================================================

with tab_2:
    tab2.design()

# ============================================================
# PubChem
# ============================================================

with tab_3:
    tab3.design()

# ============================================================
# BindingDB
# ============================================================

with tab_4:
    tab4.design()

# ============================================================
# RCSB
# ============================================================

with tab_5:
    st.info("Coming soon...")