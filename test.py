import datetime as dt
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def debug_sifma_holidays():
    try:
        from date_util import _SIFMA_
        # Check if a date is a business day
        test_date = dt.datetime(2024, 6, 19)
        assert not _SIFMA_.is_biz_day(test_date)
        assert _SIFMA_.prev_biz_day(test_date) == dt.datetime(2024, 6, 18)
        assert _SIFMA_.next_biz_day(test_date) == dt.datetime(2024, 6, 20)

        # Example usage
        st = dt.datetime(2018, 4, 1)
        et = dt.datetime(2024, 10, 9)
        days = pd.date_range(st, et, freq='D')
        days = [x for x in days if not _SIFMA_.is_biz_day(x)]
        assert len(days) == 753
        logging.info("Test passed: debug_sifma_holidays")
    except:
        logging.info("Test Failed: debug_sifma_holidays")

def debug_fomc_generation():
    try:
        from date_util import generate_fomc_meeting_dates
        st = dt.datetime(2018, 4, 1)
        et = dt.datetime(2024, 10, 9)
        fomc_meetings = generate_fomc_meeting_dates(st, et)
        assert len(fomc_meetings) == 52
        logging.info("Test passed: debug_fomc_generation")
    except:
        logging.info("Test Failed: debug_fomc_generation")

def debug_date_conversion():
    try:
        from date_util import parse_date, convert_date
        st = dt.datetime(2018, 4, 1)
        assert convert_date(st) == 41729
        assert parse_date(41729) == st

        dates = pd.date_range("2020-01-01", "2020-02-01", freq="d")
        int_array = convert_date(dates)
        assert isinstance(int_array, np.ndarray)
        assert int_array.sum() == 1356304

        dates = parse_date(int_array)
        assert isinstance(dates, np.ndarray)
        assert dates[0] == dt.datetime(2020, 1, 1)
        logging.info("Test passed: debug_date_conversion")
    except:
        logging.info("Test Failed: debug_date_conversion")


if __name__ == '__main__':
    debug_sifma_holidays()
    debug_fomc_generation()
    debug_date_conversion()

