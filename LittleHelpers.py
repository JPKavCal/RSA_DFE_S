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
        st.session_state.catch = None
        st.session_state.catchCoord = None
        st.session_state.selreg = 0
        st.session_state.selstat = 0
        st.session_state.selcoord = 0
        st.session_state.selgpt = 0
        st.session_state.catchsel = 0
        st.session_state.runcount = 0
        st.session_state.dws_prim = '../../Data Processing/0_External_Data/DWS SHP/primary catchments.shp'
        st.session_state.lastclick = 0

        st.session_state.closestdre = 0
        st.session_state.grp = 0
        st.session_state.grp_daily = 0
        st.session_state.c_dre = 0
        st.session_state.cluster = 0
        st.session_state.floods = 0

        st.set_page_config(
            page_title="Study Stations",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.session_state.initiated = 1
