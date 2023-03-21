import streamlit as st
# import os

import geopandas as gp
# import pyproj

from LittleHelpers import assemble, sidebar

import leafmap.foliumap as foliumap


@st.cache_data
def load_primary(catch_file):
    regc = gp.GeoDataFrame.from_file(catch_file)
    regc.geometry = regc.simplify(0.01)
    regc.drop('COUNT', axis=1, inplace=True)
    return regc


def reset_values():
    st.session_state.selcoord = 0
    st.session_state.catchsel = 0
    st.session_state.selgpt = 0
    st.session_state.closestdre = 0
    st.session_state.catchLenData = 0


assemble()
sidebar()

m = foliumap.Map(tiles="Stamen Terrain", latlon_control=True, search_control=False,
        measure_control=False,
        draw_control=False)

m.add_tile_layer(
    url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    name="Google Satellite",
    attribution="Google",
)

gdf = load_primary(st.session_state.dws_prim)

st.title("Simple start to zoom to primary region")

col1, col2 = st.columns(2)

reg_options = ['All', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J',
               'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
sel_type = 'Station'
with col1:
    region = st.selectbox("Select Region", reg_options, index=st.session_state.selreg)

if region == 'All':
    catch = gdf
    style_ = dict()
    reset_values()

    m.add_gdf(catch, layer_name='Primary Catchment', style=style_)
    m.zoom_to_gdf(catch)

if not st.session_state.closestdre:
    out = m.to_streamlit(height=600, bidirectional=True)
    p = out['last_object_clicked']
    p
