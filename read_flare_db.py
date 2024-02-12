import sys, os
import numpy as np
import sqlite3
# import sqlalchemy
# from sqlalchemy import create_engine, select, MetaData, Table, null

sqlite_db = {
    'path'  : '/Users/julian/Documents/phd/solar_flares/data/sql_db', # this is the path to the file
	'name'	: 'goes_flares.sqlite' # this is the name of the file
}
db_connection_string = f"sqlite:///{os.path.join(sqlite_db['path'], sqlite_db['name'])}"


def return_all_flares():
    dtypes = ['i8', 'U16', 'U16', 'U16', 'U8', 'U4', 'f8', 'f8', 'i8']
    names = ['id', 'start_time', 'peak_time', 'end_time', 'location', 'fclass', 'peak_flux', 'integ_flux', 'region_num']
    conn = sqlite3.connect(os.path.join(sqlite_db['path'], sqlite_db['name']))
    cursor = conn.cursor()
    all_flares = cursor.execute("SELECT * from flares").fetchall()
    all_flares = np.fromiter(all_flares, dtype=', '.join(dtypes))
    all_flares.dtype.names = names
    cursor.close()
    conn.close()
    return all_flares

def keep_regions_min_num(flares, min_per_region=6, verbose=False):
    regions, num_per_region = np.unique(flares['region_num'], return_counts=True)
    kept_regions = regions[num_per_region >= min_per_region]
    kept_regions = np.delete(kept_regions, np.where(kept_regions==-1)[0])
    kept_flare_indices = np.concatenate(
            [np.where(flares['region_num']==kept_region)[0] for kept_region in kept_regions]
            ).ravel()
    kept_flares = flares[kept_flare_indices]

    if verbose:
        print(f'Number of active regions with >= {min_per_region} flares: {len(kept_regions)}')
        print(f'Number of flares in these regions: {len(kept_flares)}')
    return kept_flares



if __name__=='__main__':
    # engine = create_engine(db_connection_string)
    # metadata = MetaData()

    # flares = Table('flares', metadata, autoload_with=engine)
    # regions = Table('regions', metadata, autoload_with=engine)
    # engine.dispose()
    flares = return_all_flares()
    keep_regions_min_num(flares, min_per_region=5, verbose=True)
