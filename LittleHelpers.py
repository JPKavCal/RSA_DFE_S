import streamlit as st


def sidebar():
    st.sidebar.title("About")
    st.sidebar.info(
        """
        To be completed
        """
    )

    st.sidebar.title("Contact")
    st.sidebar.info(
        """
        To be completed
        """
    )


def assemble():
    if 'initiated' not in st.session_state:
        # General variables
        st.session_state.dws_prim = 'https://drive.google.com/uc?id=1_iivRT1-Fb8wvgIPfRX2guebFD5FnlBy'

        # Study stations variables
        # st.session_state.catch = None
        st.session_state.selreg = 0
        st.session_state.selstat = 0

        # Coordinate selection variables
        st.session_state.selcoord = 0
        st.session_state.catchsel = 0
        st.session_state.selgpt = 0
        st.session_state.closestdre = 0
        st.session_state.catchLenData = 0

        st.set_page_config(
            page_title="Study Stations",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.session_state.initiated = 1
