import pandas as pd

__SOFR__FIXINGS = "fixings/SOFR.pkl"


def load_fixings(from_date, as_of):
    df = pd.read_pickle(__SOFR__FIXINGS)
    return df[from_date:as_of]


if __name__ == '__main__':
    dg = load_fixings(pd.Timestamp("2022-01-01"), pd.Timestamp("2024-9-30"))
    exit(0)
