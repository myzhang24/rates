import pandas as pd
import datetime as dt

__SOFR__FIXINGS = "fixings/SOFR.pkl"


def load_fixings(from_date: dt.datetime, as_of: dt.datetime):
    df = pd.read_pickle(__SOFR__FIXINGS)
    return df[from_date:as_of]


if __name__ == '__main__':
    dg = load_fixings(dt.datetime(2022, 1, 1), dt.datetime(2024, 10, 1))
    exit(0)
