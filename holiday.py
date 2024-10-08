"""
This module define the SIFMA instance, which supports
1. is_biz_day
2. prev_biz_day
3. next_biz_day
4. biz_day_range
queries for the SIFMA holiday calendar.
The SIFMA calendar is used to determine business days in the ModFol adjustment of SOFR OIS swap accrual start and end dates.

It also defines the NYFED instance, which generates days on which SOFR index is published.
"""

import datetime as dt
import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday, sunday_to_monday,
    USMemorialDay, USLaborDay, USMartinLutherKingJr,
    USPresidentsDay, USThanksgivingDay, GoodFriday, USColumbusDay
)


class SIFMAHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday('Juneteenth', month=6, day=19, observance=nearest_workday, start_date='2022-01-01'),
        Holiday('IndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USColumbusDay,
        Holiday('VeteransDay', month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday('ChristmasDay', month=12, day=25, observance=nearest_workday),
    ]

    # Generate Good Friday dates, excluding those on the first Friday of the month
    good_fridays = GoodFriday.dates(dt.datetime(1990, 1, 1), dt.datetime(2060, 12, 31))
    filtered_good_fridays = []
    for gf in good_fridays:
        # Get the first Friday of the month
        first_day = dt.datetime(gf.year, gf.month, 1)
        first_friday = first_day + pd.offsets.Week(weekday=4)
        # Include Good Friday if it's not the first Friday of the month (NFP publishing days)
        if gf != first_friday:
            filtered_good_fridays.append(gf)

    # Define special holidays
    special_holidays = [
                           # Hurricane Sandy closure
                           dt.datetime(2012, 10, 30),
                       ] + filtered_good_fridays


class NYFedHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday('Juneteenth', month=6, day=19, observance=sunday_to_monday, start_date='2022-01-01'),
        Holiday('IndependenceDay', month=7, day=4, observance=sunday_to_monday),
        USLaborDay,
        USColumbusDay,
        Holiday('VeteransDay', month=11, day=11, observance=sunday_to_monday),
        USThanksgivingDay,
        Holiday('ChristmasDay', month=12, day=25, observance=sunday_to_monday),
    ]


sifma_holidays = SIFMAHolidayCalendar().holidays(start='1990-01-01', end='2060-12-31')
nyfed_holidays = NYFedHolidayCalendar().holidays(start='1990-01-01', end='2060-12-31')


class Holidays:
    def __init__(self, calendar_class):
        self.holidays = calendar_class
        self.holiday_set = set(self.holidays)

    def is_biz_day(self, dt: dt.datetime | dt.date) -> bool:
        dt = pd.Timestamp(dt)
        return dt.dayofweek < 5 and dt not in self.holiday_set

    def prev_biz_day(self, dt, shift=1) -> dt.date:
        dt = pd.Timestamp(dt) - pd.Timedelta(days=shift)
        while not self.is_biz_day(dt):
            dt -= pd.Timedelta(days=1)
        return dt.to_pydatetime()

    def next_biz_day(self, dt, shift=1) -> dt.date:
        dt = pd.Timestamp(dt) + pd.Timedelta(days=shift)
        while not self.is_biz_day(dt):
            dt += pd.Timedelta(days=1)
        return dt.to_pydatetime()

    def biz_date_range(self, st, et) -> pd.Series:
        st = pd.Timestamp(st)
        et = pd.Timestamp(et)
        dates = pd.date_range(start=st, end=et, freq='D')
        biz_days = pd.to_datetime([dt for dt in dates if self.is_biz_day(dt)])
        return biz_days


# Instantiate the SIFMA and NYFED holidays
SIFMA = Holidays(sifma_holidays)
NYFED = Holidays(nyfed_holidays)

if __name__ == '__main__':
    # Check if a date is a business day
    test_date = dt.date(2024, 6, 19)
    print(f"Is {test_date} a SIFMA business day? {SIFMA.is_biz_day(test_date)}")
    print(f"Is {test_date} a NYFED business day? {NYFED.is_biz_day(test_date)}")
    print(f"Previous business day before {test_date}: {SIFMA.prev_biz_day(test_date)}")
    print(f"Next business day after {test_date}: {SIFMA.next_biz_day(test_date)}")

    # Example usage
    st = dt.datetime(2024, 1, 1)
    et = dt.datetime(2024, 2, 1)
    biz_days = SIFMA.biz_date_range(st, et)
    print(f"Business days between {st.date()} and {et.date()}:")
    for day in biz_days:
        print(day)
