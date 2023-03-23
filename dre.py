import pandas as pd
import geopandas as gp
from sqlalchemy import create_engine
from numpy import log10, power, round, array, floor, arange, int16
from itertools import product
# TODO: I HAVE TC SO DO NOT NEED ALL DURATIONS....

rps = [2, 5, 10, 20, 50, 100, 200, 500, 1000]  # is up to 1000 required?
gcs = [f"GC{x}" for x in rps]  # This excludes all upper and lower bounds, can consider for later stage
durs_m = [5, 10, 15, 30, 45, 60, 90, 120, 240, 360, 480, 600, 720, 960, 1200, 1440]
m_ar = array(durs_m)
durs_d = ['1d', '2d', '3d', '4d', '5d', '6d', '7d']
t = ['1733_1917', '1733_1916']


def calc_l(m, s, t1, t2):
    return power(10, log10(m) - s * (log10(t1) - log10(t2)))


def calc_d(m, s, t1, t2):
    return power(10, log10(m) + s * (log10(t1) - log10(t2)))


def dre_pts(poly):
    minx, miny, maxx, maxy = poly.bounds

    minx_ = int(minx) + round((minx - int(minx)) * 60, 0) / 60
    maxx_ = int(maxx) + round((maxx - int(maxx)) * 60, 0) / 60
    miny_ = int(miny) + round((miny - int(miny)) * 60, 0) / 60
    maxy_ = int(maxy) + round((maxy - int(maxy)) * 60, 0) / 60

    x_range = arange(minx_, maxx_, 1 / 60)
    y_range = arange(miny_, maxy_, 1 / 60)

    xys = [*product(x_range, y_range)]
    x = [x[0] for x in xys]
    y = [x[1] for x in xys]
    geom = gp.points_from_xy(x, y)
    # print(geom)

    gdf_poly = gp.GeoDataFrame(index=[0], geometry=[poly])
    gdf_points = gp.GeoDataFrame(index=[*range(len(geom))], geometry=geom)

    gdf_points = gdf_points.sjoin(gdf_poly)

    gdf_points['t1'] = abs(int16(
        floor(gdf_points.geometry.x) * 60 + (
                (gdf_points.geometry.x - floor(gdf_points.geometry.x)) * 60))).astype(
        str)
    gdf_points['t2'] = abs(int16(
        floor(gdf_points.geometry.y) * 60 + (
                (gdf_points.geometry.y - floor(gdf_points.geometry.y)) * 60))).astype(
        str)
    # gdf_points.to_file("./pts.shp")
    # gdf_poly.to_file("./poly.shp")
    return set((gdf_points.loc[:, "t2"] + "_" + gdf_points.loc[:, "t1"]).values)


def closest_dre_grid(lng, lat):
    deg_lng = int(lng)
    min_lng = int(round((lng - deg_lng) * 60, 0))
    deg_lat = int(lat)
    min_lat = int(round((lat - deg_lat) * 60, 0))
    return f"{abs(deg_lat) * 60 + abs(min_lat)}_{abs(deg_lng) * 60 + abs(min_lng)}", \
           (deg_lat + min_lat / 60, deg_lng + min_lng / 60)


def dre_extraction(pt_list):
    engine = create_engine("sqlite+pysqlite:///SAGrid.db")
    conn = engine.connect()
    pt_list_ = [f'"{x}"' for x in pt_list]
    # print(pt_list_)

    grdpts = pd.read_sql(
        f"SELECT ind, cluster, s_cluster, av_cluster, adj_l1_1d FROM SAGrid WHERE ind in ({', '.join(pt_list_)})",
        conn
    )

    df = None

    grdpts.set_index(['CLUSTER', 'AV_CLUSTER', 'S_CLUSTER', 'ind'], inplace=True)
    grdpts.sort_index()
    clus_comb = set(zip([*grdpts.index.get_level_values(0)],
                        [*grdpts.index.get_level_values(1)],
                        [*grdpts.index.get_level_values(2)]))

    for clus in clus_comb:
        h24 = pd.read_sql(f"SELECT * FROM '24h21dratios' WHERE cluster={clus[2]}", conn).iloc[0]

        daily = pd.read_sql(f"SELECT * FROM Daily27day WHERE region={clus[1]}", conn).iloc[0]
        temp3_1 = daily.THETA + daily.TAU * power(3, daily.SIGMA)
        temp3_2 = daily.UPSILON + daily.KAPPA * power(3, daily.RHO)

        temp7_1 = daily.THETA + daily.TAU * power(7, daily.SIGMA)
        temp7_2 = daily.UPSILON + daily.KAPPA * power(7, daily.RHO)

        gc = pd.read_sql(f"SELECT {', '.join(gcs)} FROM gc1day WHERE cluster={clus[0]}", conn).iloc[0]

        short = pd.read_sql(
            f"SELECT duration, xcoef, const FROM ShortDuration WHERE cluster={clus[2]} AND duration in (5,15,120)",
            conn, index_col='DURATION')

        for pt in grdpts.loc[clus].iterrows():
            ind = pt[0]

            l_df = pd.DataFrame()

            l1d = pt[1].ADJ_L1_1D
            l1440 = l1d * h24.MEDIAN
            l120 = l1440 * short.loc[120, 'XCOEF'] + short.loc[120, 'CONST']
            l15 = l1440 * short.loc[15, 'XCOEF'] + short.loc[15, 'CONST']
            l5 = l1440 * short.loc[5, 'XCOEF'] + short.loc[5, 'CONST']

            s1440 = (log10(l1440) - log10(l120)) / (log10(1440) - log10(120))
            s120 = (log10(l120) - log10(l15)) / (log10(120) - log10(15))
            s15 = (log10(l15) - log10(l5)) / (log10(15) - log10(5))

            l_df.loc[0, 'l5'] = l5
            l_df.loc[0, 'l10'] = calc_l(l15, s15, 15, 10)
            l_df.loc[0, 'l15'] = l15
            for d in m_ar[(m_ar > 15) & (m_ar < 120)]:
                l_df.loc[0, f'l{d}'] = calc_l(l120, s120, 120, d)

            l_df.loc[0, 'l120'] = l120
            for d in m_ar[(m_ar > 120) & (m_ar < 1440)]:
                l_df.loc[0, f'l{d}'] = calc_l(l1440, s1440, 1440, d)

            l_df.loc[0, 'l1440'] = l1440

            # Calculate daily values
            l3d = l1d * temp3_1 + temp3_2
            l7d = l1d * temp7_1 + temp7_2

            s3d = (log10(l3d) - log10(l1d)) / (log10(3) - log10(1))
            s7d = (log10(l7d) - log10(l3d)) / (log10(7) - log10(3))

            l_df.loc[0, 'l1d'] = l1d
            l_df.loc[0, 'l2d'] = calc_d(l1d, s3d, 2, 1)
            l_df.loc[0, 'l3d'] = l3d
            l_df.loc[0, 'l4d'] = calc_d(l3d, s7d, 4, 3)
            l_df.loc[0, 'l5d'] = calc_d(l3d, s7d, 5, 3)
            l_df.loc[0, 'l6d'] = calc_d(l3d, s7d, 6, 3)
            l_df.loc[0, 'l7d'] = l7d

            index = pd.MultiIndex.from_product([[ind], durs_m + durs_d], names=['pt', 'duration'])
            m_df = pd.DataFrame(round(l_df.values.reshape(23, 1) * gc.values, 1),
                                columns=rps, index=index)

            if df is None:
                df = m_df
            else:
                df = pd.concat([df, m_df])

    return df

# to get average values, then simply...
# df.groupby(['duration']).mean()


# Creation of db...
# engine2 = create_engine("sqlite+pysqlite:///SAGrid.db")
# conn = engine2.connect()
# df1 = pd.read_csv('./Daily27dayL1regressions7Region.csv', sep=';', index_col=0, decimal=',')
# df1.to_sql('Daily27day', conn)
# df1 = pd.read_csv('./24h21dratios.csv', sep=';', index_col=0, decimal=',')
# df1.to_sql('24h21dratios', conn)
# df1 = pd.read_csv('./gc1day.csv', sep=';', index_col=0, decimal=',')
# df1.to_sql('gc1day', conn)
# df1 = pd.read_csv('./ShortDurationL1Regressions.csv', sep=';', index_col=0, decimal=',')
# df1.to_sql('ShortDuration', conn)
# df1 = pd.read_csv('./SAGrid2.csv', sep=';', decimal=',')
# df1.to_sql('SAGrid', conn, if_exists='replace')