from __future__ import annotations

from pathlib import Path

import streamlit as st
import requests
import pandas as pd
import io

from src.utils.select_dir import directory_picker_siderbar, directory_picker
import src.tabs.page_data_extraction.tab2 as tab2

st.cache_data.clear()
st.cache_resource.clear()

BASE_DIR = Path(__file__).resolve().parent

# Inject custom CSS for the button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #005388; /* UniProt Blue */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 150px !important;
        height: 45px !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #003d66;
        color: white;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <h1 style="
        text-align:center;
        background: linear-gradient(90deg,#005388,#00A6D6);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        font-weight:800;
    ">
        Public Database Curation
    </h1>

    """,
    unsafe_allow_html=True,
)


tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs(
    [   
        "Uniprot",
        "ChEMBL",
        "PubChem",
        "BindingDB",
        "RCSB",
    ]
)

with tab_1:
    import src.tabs.page_data_extraction.tab1 as tab1
    tab1.design()

with tab_2:
    import src.tabs.page_data_extraction.tab2 as tab2
    tab2.design()

with tab_3:
    import src.tabs.page_data_extraction.tab3 as tab3
    tab3.design()

with tab_4:
    import src.tabs.page_data_extraction.tab4 as tab4
    tab4.design()

with tab_5:
    pass