import streamlit as st

from src.streamlit.pages.Data_Extraction.tabs.tab1 import BASE_DIR

st.cache_data.clear()
st.cache_resource.clear()


st.set_page_config(
    page_title="Cheminformatics App",
    layout="wide",
)

home_page = st.Page(
    "src/streamlit/pages/home.py",
    title="Home",
    icon=":material/home:",
)

data_extraction_page = st.Page(
    "src/streamlit/pages/Data_Extraction/data_extraction_main_page.py",
    title="Data Extraction",
    icon=":material/database:",
)
database_combine = st.Page(
    "src/streamlit/pages/Database_Combination.py",
    title="Database Combination",
    icon=":material/hub:",
)

similarity_search_page = st.Page(
    "src/streamlit/pages/Cheminformatics/Similarity_Search_Page.py",
    title="Similarity Search",
    icon=":material/search:",
)
chemical_space_page = st.Page(
    "src/streamlit/pages/Cheminformatics/Clustering_Plot_Page.py",
    title="Chemical Space",
    icon=":material/space_dashboard:",
)

machine_learning_page = st.Page(
    "src/streamlit/pages/Machine_Learning/machine_learning_main_page.py",
    title="Machine Learning",
    icon=":material/smart_toy:",
)
generative_model_page = st.Page(
    "src/streamlit/pages/Generative_Model/generative_model_main_page.py",
    title="Generative Models",
    icon="🤖",
)

predictor_page = st.Page(
    "src/streamlit/pages/Predictor.py",
    title="Activity/Property Predictor",
    icon=":material/analytics:",
)

molecular_generator_page = st.Page(
    "src/streamlit/pages/Molecular_Generator/Molecular_Generator_Main_Page.py",
    title="Molecular Generator",
    icon=":material/biotech:",
)

pg = st.navigation(
    {
        "Main": [home_page],
        "Data": [data_extraction_page, database_combine],
        "Cheminformatics":[similarity_search_page, chemical_space_page],
        "Machine Learning":[machine_learning_page, generative_model_page],
        "Predictor":[predictor_page],
        "Molecular Generator":[molecular_generator_page],
        "Molecular Optimization":[],
    }
)

pg.run()