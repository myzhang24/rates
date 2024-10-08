import pandas as pd
import numpy as np
import datetime as dt

__SOFR__FIXINGS = "fixings/SOFR.pkl"
__SOFR_FILE = "fixings/SOFR.csv"

def get_SOFR_fixings():
    df = pd.read_csv(__SOFR_FILE, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.replace(".", np.nan).dropna().astype(float)
    df.to_pickle(__SOFR__FIXINGS)
    return df

def load_SOFR_fixings(from_date: dt.datetime, as_of: dt.datetime):
    df = pd.read_pickle(__SOFR__FIXINGS)
    return df.loc[from_date:as_of, "SOFR"]


if __name__ == '__main__':
    get_SOFR_fixings()
    dg = load_SOFR_fixings(dt.datetime(2022, 1, 1), dt.datetime(2024, 10, 8))
    exit(0)
