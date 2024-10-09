import pandas as pd
import numpy as np
import datetime as dt

__fixing_cache__ = None

class FixingManager:
    def __init__(self):
        self.original_file = "fixings/SOFR.csv"
        self.clean_file = "fixings/SOFR.pkl"

    def clean_fixings(self):
        df = pd.read_csv(self.original_file, index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.replace(".", np.nan).dropna().astype(float)
        df.to_pickle(self.clean_file)
        return self

    def load_fixings(self,):
        global __fixing_cache__
        __fixing_cache__ = pd.read_pickle(self.clean_file)
        return self

    def get_fixings(self, st: dt.datetime, et: dt.datetime):
        if __fixing_cache__ is None:
            self.load_fixings()
        return __fixing_cache__[st: et]

_SOFR_ = FixingManager()

if __name__ == '__main__':

    # Clean fixings
    _SOFR_.clean_fixings()

    # Load fixings
    dg = _SOFR_.get_fixings(dt.datetime(2022, 1, 1), dt.datetime(2024, 10, 8))
    exit(0)
