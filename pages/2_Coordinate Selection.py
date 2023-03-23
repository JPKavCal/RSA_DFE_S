import streamlit as st
import os

import numpy as np
import pandas as pd
os.environ["USE_MKDOCS"] = '1'
import geopandas as gp
import pyproj

import leafmap.foliumap as leafmap
from shapely.geometry import Point, Polygon
from pysheds.grid import Grid
from shapely.ops import transform, nearest_points
# from osgeo import gdal
from rasterio import Affine
from rasterio import open as rasopen
from rasterio.mask import mask

from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.palettes import Category10_9

import dre

from LittleHelpers import assemble, sidebar


@st.cache_data
def calc_floods(clust, area, map, dc):
    cluster_gfs = f"./regions/tables/cluster_gfs.csv"
    cluster_pars = f"./regions/tables/cluster_calc_gf.csv"

    gfs = pd.read_csv(cluster_gfs, index_col=0)
    pars = pd.read_csv(cluster_pars, index_col=0)

    ar = np.log(np.array([np.e, area, map, dc]))

    maf = np.exp(sum(ar * pars.loc[clust].values))

    return maf * gfs.loc[clust], maf


@st.cache_data
def delineate_catch(_gridl, _fdirl, x, y):
    wgs84 = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
    aea = '+proj=aea +lat_0=0 +lon_0=25 +lat_1=20 +lat_2=-23 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'

    # Delineate the catchment
    catchl = _gridl.catchment(x=x, y=y, dirmap=(3, 2, 1, 8, 7, 6, 5, 4), fdir=_fdirl,
                              xytype='coordinate', snap="center")

    _gridl.clip_to(catchl)

    # Create view
    catch_view = _gridl.view(catchl, dtype=np.uint8)

    # Create a vector representation of the catchment mask
    shapes = _gridl.polygonize(catch_view)
    polys = [(len(shape['coordinates'][0]), Polygon(shape['coordinates'][0])) for shape, value in shapes]

    biggest = polys[0]
    if len(polys) > 1:
        for i in range(len(polys)):
            if polys[i][0] > biggest[0]:
                biggest = polys[i]

    project = pyproj.Transformer.from_crs(wgs84, aea, always_xy=True).transform
    aea_poly = transform(project, biggest[1])

    map_path = f"./regions/map/RSA_MAP.tif"

    with rasopen(map_path) as src:
        map_vals, _ = mask(src, [biggest[1]], crop=True)
        map_mean = int(map_vals[map_vals != src.nodata].mean().round())

    return biggest[1], np.round(aea_poly.area/1000000, 3), np.round(aea_poly.length / 1000, 3), map_mean


@st.cache_data
def load_primary(catch_file):
    regc = gp.GeoDataFrame.from_file(catch_file)
    regc.geometry = regc.simplify(0.01)
    regc.drop('COUNT', axis=1, inplace=True)
    return regc


def find_closest_grid_cell(gridp, pt):
    t0 = gridp.affine
    t1 = t0 * Affine.translation(0.5, 0.5)
    return t1 * (pt[0], pt[1])


@st.cache_data
def get_cluster(coord):
    t = Point(coord)
    clusters = gp.GeoDataFrame.from_file("./regions/shp/Clusters.shp")
    return clusters[clusters.contains(t)].Cluster.values[0]


# @st.cache_data
def dist_to_coast(_pt):
    cl = gp.read_file('./regions/shp/Southern_Africa_Coast.shp')
    pp = nearest_points(_pt, cl.geometry[0])
    return np.sqrt((pp[1].coords[0][0] - pp[0].coords[0][0]) ** 2 + (pp[1].coords[0][1] - pp[0].coords[0][1]) ** 2)


# @st.cache_data
# def longest_flow_path(cell, region_):
#     """
#     Delineates the longest flow path and adds various attributes
#
#     """
#     import numpy as np
#     from shapely import LineString
#     from shapely.ops import transform
#     import pyproj
#
#     fd_list = [8, 1, 2, 7, 3, 6, 5, 4]
#     shift_list = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
#     cell = [int(x) for x in cell]
#     cell_list = []
#     vertex_list = []
#     length_list = [0]
#     el_list = []
#     area_list = [0]
#
#     wgs84 = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
#     aea = '+proj=aea +lat_0=0 +lon_0=25 +lat_1=20 +lat_2=-23 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'
#     project = pyproj.Transformer.from_crs(wgs84, aea, always_xy=True).transform
#
#     map_path = f"./regions/map/RSA_MAP.tif"
#     el_path = f"./regions/Fel/Region_{region_}_fel.tif"
#     fd_path = f"./regions/FD/Region_{region_}_FD.tif"
#     path = f"./regions/FAcc/Region_{region_}_FAcc.tif"
#
#     r = gdal.Open(path)
#     band = r.GetRasterBand(1)
#
#     f_dr = gdal.Open(fd_path)
#     f_dband = f_dr.GetRasterBand(1)
#
#     el_ras = gdal.Open(el_path)
#     el_band = el_ras.GetRasterBand(1)
#
#     (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = r.GetGeoTransform()
#
#     past_point = band.ReadAsArray(cell[0], cell[1], 1, 1)[0][0]
#
#     # add first point to line
#     x = cell[0] * x_size + upper_left_x + (x_size / 2)  # add half the cell size
#     y = cell[1] * y_size + upper_left_y + (y_size / 2)  # to centre the point
#
#     vertex_list.append([np.round(x, 4), np.round(y, 4)])
#     cell_list.append(cell)
#     el_list.append(el_band.ReadAsArray(cell[0], cell[1], 1, 1)[0][0])
#
#     while past_point > 0:
#         # Setup for remaining checks and flow path tracing
#         bounding_cells = []
#         bounding_values = []
#         bounding_fd = []
#
#         for shift in shift_list:
#             try:
#                 bounding_fd.append(f_dband.ReadAsArray(shift[1] + cell[0], shift[0] + cell[1], 1, 1)[0][0])
#                 bounding_values.append(band.ReadAsArray(shift[1] + cell[0], shift[0] + cell[1], 1, 1)[0][0])
#                 bounding_cells.append([shift[1] + cell[0], shift[0] + cell[1]])
#             except:
#                 continue
#
#         pop_list = []
#
#         for value in range(len(bounding_values)):
#             if bounding_values[value] > past_point:
#                 pop_list.append(value)
#             if bounding_values[value] == past_point and bounding_fd[value] != fd_list[value]:
#                 pop_list.append(value)
#
#         pop_list.reverse()
#         for item in pop_list:
#             bounding_values.pop(item)
#             bounding_cells.pop(item)
#
#         past_point = max(set(bounding_values))
#         cell = [bounding_cells[bounding_values.index(past_point)][0],
#                 bounding_cells[bounding_values.index(past_point)][1]]
#
#         x = cell[0] * x_size + upper_left_x + (x_size / 2)  # add half the cell size
#         y = cell[1] * y_size + upper_left_y + (y_size / 2)  # to centre the point
#
#         vertex_list.append([np.round(x, 4), np.round(y, 4)])
#         cell_list.append(cell)
#         el_list.append(el_band.ReadAsArray(cell[0], cell[1], 1, 1)[0][0])
#
#     line_ = LineString(vertex_list)
#
#     el_drop_list = list(map(lambda l: abs(l - el_list[0]), el_list))
#
#     line = transform(project, line_)
#     length = line.length
#
#     a2 = [*line.coords]
#
#     for i in range(1, len(vertex_list)):
#         length_list.append(length_list[i - 1] + np.sqrt((a2[i][0] - a2[i - 1][0]) ** 2
#                                                         + (a2[i][1] - a2[i - 1][1]) ** 2))
#         area_list.append(area_list[i - 1] + el_drop_list[i - 1] * (length_list[i] - length_list[i - 1]) +
#                          0.5 * (length_list[i] - length_list[i - 1]) * (
#                                  el_drop_list[i] - el_drop_list[i - 1]))
#
#     sea = (2 * area_list[-1] / length) / length
#     d10 = list(map(lambda x: abs(x - length_list[-1] * .1), length_list))
#     d85 = list(map(lambda x: abs(x - length_list[-1] * .85), length_list))
#
#     d10_loc = d10.index(min(d10))
#     d85_loc = d85.index(min(d85))
#
#     s_cell = [int(round((vertex_list[0][0] - upper_left_x - (x_size / 2)) / x_size, 0)),
#               int(round((vertex_list[0][1] - upper_left_y - (y_size / 2)) / y_size, 0))]
#
#     d10_cell = [int(round((vertex_list[d10_loc][0] - upper_left_x - (x_size / 2)) / x_size, 0)),
#                 int(round((vertex_list[d10_loc][1] - upper_left_y - (y_size / 2)) / y_size, 0))]
#
#     d85_cell = [int(round((vertex_list[d85_loc][0] - upper_left_x - (x_size / 2)) / x_size, 0)),
#                 int(round((vertex_list[d85_loc][1] - upper_left_y - (y_size / 2)) / y_size, 0))]
#
#     e_cell = [int(round((vertex_list[-1][0] - upper_left_x - (x_size / 2)) / x_size, 0)),
#               int(round((vertex_list[-1][1] - upper_left_y - (y_size / 2)) / y_size, 0))]
#
#     d10_val = el_band.ReadAsArray(d10_cell[0], d10_cell[1], 1, 1)[0]
#     d85_val = el_band.ReadAsArray(d85_cell[0], d85_cell[1], 1, 1)[0]
#     s_val = el_band.ReadAsArray(s_cell[0], s_cell[1], 1, 1)[0]
#     e_val = el_band.ReadAsArray(e_cell[0], e_cell[1], 1, 1)[0]
#
#     del el_band
#
#     s10 = (d85_val - d10_val) / (0.75 * length)
#
#     return line_, {
#         "Length (km)": np.round(length / 1000, 3),
#         "Elevation (Start)": int(s_val[0]),
#         "Elevation (10%)": int(d10_val[0]),
#         "Elevation (85%)": int(d85_val[0]),
#         "Elevation (End)": int(e_val[0]),
#         "Slope": (float(e_val[0]) - float(s_val[0])) / length,
#         "Slope (Equal Area)": float(sea),
#         "Slope (10-85)": float(s10[0]),
#         "Tc (hr)": np.round(((0.87 * (length / 1000) ** 2) / (1000 * float(s10[0]))) ** 0.385, 2)
#     }


def reset_values():
    st.session_state.selcoord = 0
    st.session_state.catchsel = 0
    st.session_state.selgpt = 0
    st.session_state.closestdre = 0
    st.session_state.catchLenData = 0


assemble()
sidebar()

m = leafmap.Map(tiles="Stamen Terrain", latlon_control=True, search_control=False,
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
else:
    if reg_options.index(region) != st.session_state.selreg:
        reset_values()

    rast = f"./regions/FAcc_1000/{region}_Thresh_1000_byte2.tif"
    if os.path.exists(rast):
        m.add_raster(rast, 1, nodata=0, palette="BuGn")

    style_ = {"stroke": True,
              "color": "#05f711",
              "weight": 4,
              "opacity": 1,
              "fill": True,
              "fillColor": "#ffffff",
              "fillOpacity": 0.1,
              "clickable": True
              }

    flow_style_ = {"stroke": True,
                   "color": "#0722f0",
                   "weight": 4,
                   "opacity": 1,
                   "clickable": True
                   }

    if type(st.session_state.selcoord) == int:
        with col2:
            st.write("Select the desired point on the map")
        st.session_state.catch = None
        catch = gdf[gdf.PRIMARY == region].copy()
        m.add_gdf(catch, layer_name='Primary Catchment', style=style_)
        m.zoom_to_gdf(catch)
    else:

        coords = st.session_state.selcoord
        m.add_marker([coords[1], coords[0]])

        with col1:
            if st.button("Reset"):
                reset_values()
                st.experimental_rerun()

        if not st.session_state.catchsel:
            grid = Grid.from_raster(f"./regions/FD/Region_{region}_FD.tif")
            fdir = grid.read_raster(f"./regions/FD/Region_{region}_FD.tif")

            st.session_state.catchData = delineate_catch(grid, fdir,
                                                         coords[0],
                                                         coords[1])

            # st.session_state.catchLenData = longest_flow_path(st.session_state.selgpt,
            #                                                   region)

            st.session_state.dcoast = dist_to_coast(Point([coords[0], coords[1]]))

            st.session_state.floods = calc_floods(st.session_state.cluster,
                                                  st.session_state.catchData[1],
                                                  st.session_state.catchData[3],
                                                  st.session_state.dcoast)

            with st.spinner('Extracting rainfall'):
                drepts = dre.dre_pts(st.session_state.catchData[0])
                dre_df = dre.dre_extraction(drepts)

                if dre_df is None:
                    dre_df = dre.dre_extraction([st.session_state.closestdre[0]])

                st.session_state.grp = dre_df.groupby("duration").mean()
                st.session_state.grp_daily = st.session_state.grp.iloc[16:]
                st.session_state.grp_daily.index = [*range(1, 8)]
                st.session_state.grp_daily.index.name = "duration"
                st.session_state.grp = st.session_state.grp.iloc[:16]
                # tc = st.session_state.catchLenData[1]["Tc (hr)"]

                # if st.session_state.catchLenData[1]["Tc (hr)"] < 24:
                #     st.session_state.grp.loc[
                #         st.session_state.catchLenData[1]["Tc (hr)"] * 60] = np.nan
                #     st.session_state.grp.sort_index(inplace=True)
                #     st.session_state.grp.interpolate(method='index', inplace=True)
                #     st.session_state.c_dre = st.session_state.grp.loc[
                #         st.session_state.catchLenData[1]["Tc (hr)"] * 60]
                #
                # elif st.session_state.catchLenData[1]["Tc (hr)"] < 168:
                #     st.session_state.grp_daily.loc[
                #         st.session_state.catchLenData[1]["Tc (hr)"] / 24] = np.nan
                #     st.session_state.grp_daily.sort_index(inplace=True)
                #     st.session_state.grp_daily.interpolate(method='index', inplace=True)
                #     st.session_state.c_dre = st.session_state.grp_daily.loc[
                #         st.session_state.catchLenData[1]["Tc (hr)"] / 24]
                # TODO: ADD CLAUSE IF 7 days exceeded...

                st.session_state.grp.reset_index(inplace=True)
                st.session_state.grp_daily.reset_index(inplace=True)

            st.session_state.catchsel = 1

        df = pd.DataFrame({"Outlet Latitude": [coords[1]],
                           "Outlet Longitude": [coords[0]],
                           "Area (km2)": [st.session_state.catchData[1]],
                           "Perimeter (km)": [st.session_state.catchData[2]],
                           "MAP mean (mm)": [st.session_state.catchData[3]],
                           "Distance from Coast (Decimal Degree)": st.session_state.dcoast})
        # df2 = pd.DataFrame(st.session_state.catchLenData[1], index=[0])
        df.set_index("Outlet Latitude", inplace=True)
        # df2.set_index("Length (km)", inplace=True)
        # with col2:

        # m.add_geojson(st.session_state.catchLenData[0].__geo_interface__,
        #               layer_name='Flow Length',
        #               info_mode=None,
        #               style=flow_style_)

        # feature = {"type": "FeatureCollection",
        #            "features": [{'type': 'Feature',
        #                          "geometry": st.session_state.catchLenData[0].__geo_interface__,
        #                          "properties": st.session_state.catchLenData[1]
        #                          }]}

        m.add_geojson(st.session_state.catchData[0].__geo_interface__,
                      layer_name='Catchment',
                      info_mode=None,
                      style=style_)

        m.zoom_to_bounds(st.session_state.catchData[0].bounds)

        sd_fig = figure(title="Design Rainfall Depths (Sub-daily)",  # x_axis_type="log",
                        x_range=(5, 1440),
                        background_fill_color="#fafafa",
                        # toolbar_location=None,
                        # tools=[HoverTool()],
                        tooltips="@x minutes - @y mm",
                        toolbar_location="right"
                        )

        for rp, color in zip([2, 5, 10, 20, 50, 100, 200, 500, 1000], Category10_9):
            sd_fig.line(st.session_state.grp.loc[:, 'duration'], st.session_state.grp.loc[:, rp],
                        line_width=2, color=color, alpha=0.8,
                        legend_label=f"{1 / rp}% AEP")
        sd_fig.legend[0].items.reverse()
        sd_fig.legend.location = "top_left"
        sd_fig.legend.click_policy = "hide"

        sd_fig.xaxis[0].axis_label = 'Storm Duration (min)'
        sd_fig.yaxis[0].axis_label = 'Rainfall Depth (mm)'

        d_fig = figure(title="Design Rainfall Depths (Daily)",  # x_axis_type="log",
                       x_range=(1, 7),
                       background_fill_color="#fafafa",
                       # toolbar_location=None,
                       # tools=[HoverTool()],
                       tooltips="@x day/s - @y mm",
                       toolbar_location="right"
                       )

        for rp, color in zip([2, 5, 10, 20, 50, 100, 200, 500, 1000], Category10_9):
            d_fig.line(st.session_state.grp_daily.loc[:, 'duration'], st.session_state.grp_daily.loc[:, rp],
                       line_width=2, color=color, alpha=0.8,
                       legend_label=f"{1 / rp}% AEP")
        d_fig.legend[0].items.reverse()
        d_fig.legend.location = "top_left"
        d_fig.legend.click_policy = "hide"

        d_fig.xaxis[0].axis_label = 'Storm Duration (min)'
        d_fig.yaxis[0].axis_label = 'Rainfall Depth (mm)'

        aeps = 1 / st.session_state.c_dre.index.values * 100
        # c_fig = figure(title="Design Rainfall Depths (Sub-daily)", x_axis_type="log",
        #                x_range=(aeps.max(), aeps.min()),
        #                background_fill_color="#fafafa",
        #                tooltips="@x % AEP - @y mm",
        #                toolbar_location="right"
        #                )
        # c_fig.add_tools(HoverTool(
        #     tooltips=[
        #         ("AEP", '@x % AEP'),
        #         ("Design Rainfall", '@y mm'),
        #     ],
        #     mode='vline'
        # ))
        # c_fig.line(aeps, st.session_state.c_dre.values,
        #            line_width=2, color=color, alpha=0.8,
        #            legend_label=f"Tc = {st.session_state.catchLenData[1]['Tc (hr)']} hours")
        #
        # c_fig.legend[0].items.reverse()
        # c_fig.legend.location = "top_left"
        # c_fig.legend.click_policy = "hide"
        #
        # c_fig.xaxis[0].axis_label = 'Annual Exceedance Probability (%)'
        # c_fig.yaxis[0].axis_label = 'Rainfall Depth (mm)'

        pf_fig = figure(title="Regional Design Peak Flow", x_axis_type="log",
                       x_range=(aeps.max(), aeps[6]),
                       background_fill_color="#fafafa",
                       tooltips="@x % AEP - @y m3.s-1",
                       toolbar_location="right"
                       )
        pf_fig.add_tools(HoverTool(
            tooltips=[
                ("AEP", '@x % AEP'),
                ("Peak Flow", '@y m3.s-1'),
            ],
            mode='vline'
        ))

        # Random Rational method calc
        # c_vals = np.random.random(7)
        # c_vals.sort()
        # i_vals = st.session_state.c_dre.values[:7] / st.session_state.catchLenData[1]["Tc (hr)"]
        # rat_q = st.session_state.catchData[1] * c_vals * i_vals / 3.6

        pf_fig.line(aeps[:7], st.session_state.floods[0].values,
                    line_width=2, color=color, alpha=0.8,
                    legend_label=f"Regional model"
                    )

        pf_fig.legend.location = "top_left"
        pf_fig.legend.click_policy = "hide"

        pf_fig.xaxis[0].axis_label = 'Annual Exceedance Probability (%)'
        pf_fig.yaxis[0].axis_label = 'Design Peak Flow (m2.s-1)'

if not st.session_state.closestdre:
    out = m.to_streamlit(height=600, bidirectional=True)
    p = out['last_object_clicked']
else:
    mcol, dfcol = st.columns(2)

    with st.expander("Interactive Map"):
        m.to_streamlit(height=600, bidirectional=True)
        # st.download_button("Download Flow Path",
        #                    feature.__repr__().replace("'", '"').replace("(", "[").replace(")", "]"),
        #                    file_name="flow.json")
    p = None

    with st.expander("Catchment Attributes"):
        st.dataframe(df)
        # st.dataframe(df2)

    with st.expander("Rainfall Data"):
        tab1, tab2, tab3 = st.tabs(["Catchment", "Sub-daily", "Daily"])
        with tab1:
            pass
            # t1c1, t1c2 = st.columns(2)
            # with t1c1:
            #     # st.bokeh_chart(c_fig, use_container_width=True)
            # with t1c2:
            #     st.dataframe(st.session_state.c_dre)
            #     st.download_button("Download Catchment Rainfall",
            #                        st.session_state.c_dre.to_csv(),
            #                        file_name="catch_rain.csv")
        with tab2:
            t2c1, t2c2 = st.columns(2)
            with t2c1:
                st.bokeh_chart(sd_fig, use_container_width=True)
            with t2c2:
                st.dataframe(st.session_state.grp.set_index("duration"))
                st.download_button("Download Catchment Sub Daily Rainfall",
                                   st.session_state.grp.to_csv(),
                                   file_name="catch_subdaily_rain.csv")
        with tab3:
            t3c1, t3c2 = st.columns(2)
            with t3c1:
                st.bokeh_chart(d_fig, use_container_width=True)
            with t3c2:
                st.dataframe(st.session_state.grp_daily.set_index("duration"))
                st.download_button("Download Catchment Daily Rainfall",
                                   st.session_state.grp_daily.to_csv(),
                                   file_name="catch_daily_rain.csv")

    with st.expander("Peak Flows"):

        pfc1, pfc2 = st.columns(2)
        with pfc1:
            st.bokeh_chart(pf_fig, use_container_width=True)
        with pfc2:
            st.write(f"Model Cluster - {st.session_state.cluster}")
            st.write(f"Mean Annual Flood - {np.round(st.session_state.floods[1], 4)}")
            st.session_state.floods[0]


if p is not None and type(st.session_state.selcoord) == int:

    if region == 'All':
        cpl = catch.sindex.nearest(Point(p['lng'],
                                         p['lat']), return_distance=True)
        cp = cpl[0][1][0]
        try:
            st.session_state.selreg = reg_options.index(catch.iloc[cp].PRIMARY)
            st.experimental_rerun()
        except ValueError:
            pass
    else:
        grid = Grid.from_raster(rast)
        gthresh = grid.read_raster(rast)
        ptmp = grid.snap_to_mask(gthresh,
                                 [[p['lng'],
                                   p['lat']]])[0]

        st.session_state.lastclick = ptmp
        ptmp = grid.nearest_cell(ptmp[0],
                                 ptmp[1],
                                 snap="center")
        if gthresh[ptmp[1], ptmp[0]] != 1:
            # print("ADJUSTING")
            ptmp = [ptmp[0], ptmp[1] + 1]
            if gthresh[ptmp[1], ptmp[0]] != 1:
                ptmp = [ptmp[0] + 1, ptmp[1]]

        st.session_state.selgpt = ptmp
        st.session_state.selcoord = find_closest_grid_cell(grid, ptmp)

        st.session_state.selreg = reg_options.index(region)
        st.session_state.closestdre = dre.closest_dre_grid(p['lng'], p['lat'])
        st.session_state.cluster = get_cluster(st.session_state.selcoord)
        st.experimental_rerun()
