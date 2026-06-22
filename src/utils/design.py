import streamlit as st 
import time

def thick_divider():
    st.markdown("<hr>", unsafe_allow_html=True)

def temp_success(msg, seconds=3):
    placeholder = st.empty()

    with placeholder.container():
        st.success(msg)

    time.sleep(seconds)
    placeholder.empty()

def temp_info(msg, seconds=3):
    placeholder = st.empty()

    with placeholder.container():
        st.info(msg)

    time.sleep(seconds)
    placeholder.empty()

def temp_error(msg, seconds=3):
    placeholder = st.empty()

    with placeholder.container():
        st.error(msg)

    time.sleep(seconds)
    placeholder.empty()