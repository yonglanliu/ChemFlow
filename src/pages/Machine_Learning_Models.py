# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from src.utils.design import temp_success, temp_info

st.markdown(
    """
    <h1 style="
        text-align:center;
        background: linear-gradient(90deg,#005388,#00A6D6);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        font-weight:800;
    ">
        Traditional Machine Learning Models
    </h1>
    """,
    unsafe_allow_html=True,
)

st.divider()

tab_1, tab_2, tab_3 = st.tabs(
    [   
        "Setup",
        "Analysis",
        "Plot Data"
    ]
)

with tab_1:
    from src.streamlit.tabs.page_machine_learning import tab1_setup
    workdir = tab1_setup.design()

with tab_2:
    from src.streamlit.tabs.page_machine_learning import tab2_analysis
    tab2_analysis.design(workdir)

with tab_3:
    pass