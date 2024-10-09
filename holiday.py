"""
This module define the _SIFMA_ instance, which supports
1. is_biz_day
2. prev_biz_day
3. next_biz_day
4. biz_day_range
queries for the SIFMA holiday calendar.
The SIFMA calendar is used to determine business days in the ModFol adjustment of SOFR OIS swap accrual start and end dates.
"""

import datetime as dt
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    nearest_workday,
    USMemorialDay,
    USLaborDay,
    USMartinLutherKingJr,
    USPresidentsDay,
    USThanksgivingDay,
    GoodFriday,
    USColumbusDay
)


class SIFMACalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('Juneteenth', month=6, day=19, observance=nearest_workday, start_date='2022-01-01'),
        Holiday('IndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USColumbusDay,
        Holiday('VeteransDay', month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday('ChristmasDay', month=12, day=25, observance=nearest_workday),
    ]

    def __init__(self):
        super().__init__()
        self.holiday_set = self.holidays(start='1990-01-01', end='2060-12-31')

    def is_biz_day(self, d: dt.datetime | dt.date) -> bool:
        return d.weekday() < 5 and d not in self.holiday_set

    def prev_biz_day(self, d: dt.datetime | dt.date, shift=1) -> dt.datetime | dt.date:
        d -= dt.timedelta(days=shift)
        while not self.is_biz_day(d):
            d -= dt.timedelta(days=1)
        return d

    def next_biz_day(self, d: dt.datetime | dt.date, shift=1) -> dt.datetime | dt.date:
        d += dt.timedelta(days=shift)
        while not self.is_biz_day(d):
            d -= dt.timedelta(days=1)
        return d

    def biz_date_range(self, st: dt.datetime | dt.date, et: dt.datetime | dt.date) -> pd.Series:
        dates = pd.date_range(start=st, end=et, freq='D')
        biz_days = pd.to_datetime([dt for dt in dates if self.is_biz_day(dt)])
        return biz_days

_SIFMA_ = SIFMACalendar()

if __name__ == '__main__':
    # Check if a date is a business day
    test_date = dt.date(2024, 6, 19)
    print(f"Is {test_date} a SIFMA business day? {_SIFMA_.is_biz_day(test_date)}")
    print(f"Previous business day before {test_date}: {_SIFMA_.prev_biz_day(test_date)}")
    print(f"Next business day after {test_date}: {_SIFMA_.next_biz_day(test_date)}")

    # Example usage
    st = dt.datetime(2018, 4, 1)
    et = dt.datetime(2024, 10, 9)
    days = pd.date_range(st, et, freq='D')
    for day in days:
        if not _SIFMA_.is_biz_day(day):
            if day.dayofweek < 5:
                print(f"{day.date()} is not a SIFMA business day")
