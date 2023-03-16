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
    # Caching function for loading the regional stations into a geopandas dataframe
    stats = gp.GeoDataFrame.from_file(station_file)
    stats['lon'] = stats.long
    regc = gp.GeoDataFrame.from_file(catch_file)
    stats.crs = regc.crs
    return stats, regc


@st.cache_data  # (allow_output_mutation=True)
def load_primary(catch_file):
    # Caching function for loading the DWS Primary drainage regions into a geopandas dataframe
    regc = gp.GeoDataFrame.from_file(catch_file)
    regc.geometry = regc.simplify(0.01)
    regc.drop('COUNT', axis=1, inplace=True)
    return regc


# Link to google drive as test of speed
dws_prim = 'https://drive.google.com/uc?id=1_iivRT1-Fb8wvgIPfRX2guebFD5FnlBy'
gdf = load_primary(dws_prim)

st.title("Study Stations and Catchments")

col1, col2 = st.columns(2)

# Define list of DWS primary drainage regions, could also extract from gdf...
reg_options = ['All', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J',
               'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
# sel_type = 'Station'

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
    else:  # if sel_type == 'Station':
        st.session_state.selstat = options.index(station)
        stat = stations[stations['gauge'] == station]
        catch = regcatch[regcatch['gauge'] == station].loc[:, ['gauge', 'Area', 'Lc', 'ARF', 'Tc', 'geometry']]
        catch = np.round(catch, 2)

        st.session_state.catch = regcatch[regcatch['gauge'] == station].drop('geometry', axis=1)

else:
    catch = gdf
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

if region != 'All' and st.session_state.selstat == 0:  # and sel_type == 'Station':
    m.add_gdf(stat, layer_name='Study Stations')
elif st.session_state.selstat != 0:
    m.add_marker((stat.iloc[0].geometry.y, stat.iloc[0].geometry.x))

height = 600
m.add_gdf(catch, layer_name='Primary Catchment')
m.zoom_to_gdf(catch)

if st.session_state.selstat == 0:

    out = m.to_streamlit(height=height, bidirectional=True)
    p = out['last_object_clicked']

    if p is not None:
        if region != 'All' and st.session_state.selstat == 0:
            cpl = stations.sindex.nearest(Point(p['lng'], p['lat']), return_distance=True)
            cp = cpl[0][1][0]
            dist = cpl[1][0]
            if dist == 0:
                st.session_state.selstat = options.index(stat.iloc[cp].gauge)
                st.experimental_rerun()
        elif region == 'All':
            cpl = catch.sindex.nearest(Point(p['lng'], p['lat']), return_distance=True)
            cp = cpl[0][1][0]
            try:
                st.session_state.selreg = reg_options.index(catch.iloc[cp].PRIMARY)
                st.experimental_rerun()
            except ValueError:
                st.write("Error here...")
                pass

else:
    stcol1, stcol2 = st.columns(2)
    with stcol1:
        m.to_streamlit(height=height)

    with stcol2:
        stat.drop('geometry', axis=1, inplace=True)
        catch.drop('geometry', axis=1, inplace=True)
        st.dataframe(stat)
        st.dataframe(catch)
