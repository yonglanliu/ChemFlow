import streamlit as st

st.cache_data.clear()
st.cache_resource.clear()


st.set_page_config(
    page_title="Cheminformatics App",
    layout="wide",
)

home_page = st.Page(
    "src/pages/home.py",
    title="Home",
    icon=":material/home:",
)

data_extraction_page = st.Page(
    "src/pages/Data_Extraction.py",
    title="Data Extraction",
    icon=":material/database:",
)
database_combine = st.Page(
    "src/pages/Database_Combination.py",
    title="Database Combination",
    icon=":material/hub:",
)

machine_learning_page = st.Page(
    "src/pages/Machine_Learning_Models.py",
    title="Machine Learning",
    icon=":material/smart_toy:",
)

predictor_page = st.Page(
    "src/pages/Predictor.py",
    title="Activity/Property Predictor",
    icon=":material/analytics:",
)

molecular_generator_page = st.Page(
    "src/pages/Molecular_Generator.py",
    title="Molecular Generator",
    icon=":material/biotech:",
)

pg = st.navigation(
    {
        "Main": [home_page],
        "Data": [data_extraction_page, database_combine],
        "Clustering":[],
        "Machine Learning":[machine_learning_page],
        "Predictor":[predictor_page],
        "Molecular Generator":[molecular_generator_page],
        "Molecular Optimization":[],
    }
)

pg.run()