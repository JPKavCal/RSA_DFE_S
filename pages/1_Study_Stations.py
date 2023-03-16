import geopandas as gp
import streamlit as st
import numpy as np
import leafmap.foliumap as leafmap
from shapely.geometry import Point

from LittleHelpers import assemble, sidebar

assemble()
sidebar()


@st.cache_data  # (allow_output_mutation=True)
def load_stations(station_file, catch_file):
    stats = gp.GeoDataFrame.from_file(station_file)
    stats['lon'] = stats.long
    regc = gp.GeoDataFrame.from_file(catch_file)
    stats.crs = regc.crs
    return stats, regc


@st.cache_data  # (allow_output_mutation=True)
def load_primary(catch_file):
    regc = gp.GeoDataFrame.from_file(catch_file)
    regc.geometry = regc.simplify(0.01)
    regc.drop('COUNT', axis=1, inplace=True)
    return regc


# primcatch_path = '../../Data Processing/0_External_Data/DWS SHP/primary catchments.shp'
# gdf = gp.read_file(primcatch_path)

dws_prim = 'https://drive.google.com/uc?id=1_iivRT1-Fb8wvgIPfRX2guebFD5FnlBy'
# dws_prim = '../../Data Processing/0_External_Data/DWS SHP/primary catchments.shp'
gdf_dws_prim = load_primary(dws_prim)
gdf = gdf_dws_prim.copy()

st.title("Study Stations and Catchments")

col1, col2 = st.columns(2)

reg_options = ['All', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J',
               'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
sel_type = 'Station'
with col1:
    region = st.selectbox("Select Region", reg_options, index=st.session_state.selreg)
    st.session_state.selreg = reg_options.index(region)

# container = st.container()
station = 'All'

if region != 'All':
    stations, regcatch = load_stations(f'./regions/Stations/{region}_Stations.shp',
                                       f'./regions/Catchments/WSPop_{region}.shp')

    with col2:
        options = ["All"] + list(stations.gauge.sort_values())
        station = st.selectbox('Station', options, index=st.session_state.selstat)

    if station == 'All':
        stat = stations.copy()
        catch = gdf[gdf.PRIMARY == region].copy()
        st.session_state.catch = None
        st.session_state.selstat = 0
    elif sel_type == 'Station':
        st.session_state.selstat = options.index(station)
        stat = stations[stations['gauge'] == station]
        catch = regcatch[regcatch['gauge'] == station].loc[:, ['gauge', 'Area', 'Lc', 'ARF', 'Tc', 'geometry']]
        catch = np.round(catch, 2)

        st.session_state.catch = regcatch[regcatch['gauge'] == station].drop('geometry', axis=1)
    else:
        # st.session_state.selstat = 0
        st.session_state.catch = None
        catch = gdf[gdf.PRIMARY == region].copy()


else:
    catch = gdf_dws_prim
    st.session_state.selstat = 0

lon, lat = leafmap.gdf_centroid(catch)

m = leafmap.Map(center=(lat, lon), tiles="Stamen Terrain", latlon_control=True, search_control=False,
                measure_control=False,
                draw_control=False)
m.add_tile_layer(
    url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    name="Google Satellite",
    attribution="Google",
)


if region != 'All' and sel_type == 'Station':
    m.add_gdf(stat, layer_name='Study Stations')

m.add_gdf(catch, layer_name='Primary Catchment')
m.zoom_to_gdf(catch)

height = 600
out = m.to_streamlit(height=height, bidirectional=True)
# out
p = out['last_object_clicked']

if p is not None:
    if region != 'All' and sel_type == 'Station':
        cpl = stations.sindex.nearest(Point(p['lng'], p['lat']), return_distance=True)
        cp = cpl[0][1][0]
        dist = cpl[1][0]
        if dist == 0:
            st.session_state.selstat = options.index(stat.iloc[cp].gauge)
            st.experimental_rerun()
    elif region != 'All' and sel_type != 'Station':
        pass
    else:
        cpl = catch.sindex.nearest(Point(p['lng'], p['lat']), return_distance=True)
        cp = cpl[0][1][0]
        try:
            st.session_state.selreg = reg_options.index(catch.iloc[cp].PRIMARY)
            st.experimental_rerun()
        except ValueError:
            pass
