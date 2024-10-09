"""
This module generates the swap schedule for a USD SOFR OIS swap with market convention.
Roll convention of End of Month and IMM are supported.
"""

import datetime as dt
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
from holiday import SIFMA


def fra_start_end_date(ref_date, short_hand) -> (dt.date, dt.date):
    ref_date = pd.Timestamp(ref_date)
    a, b = [int(x) for x in short_hand.split("x")]
    unadj_start = ref_date + pd.DateOffset(months=a)
    unadj_end = ref_date + pd.DateOffset(months=b)
    return unadj_start.as_pydatetime(), unadj_end.as_pydatetime()


def adjust_date(date, convention):
    if convention == 'Following':
        adjusted_date = SIFMA.next_biz_day(date, 0)
    elif convention == 'Modified Following':
        adjusted_date = modified_following(date)
    elif convention == 'Preceding':
        adjusted_date = SIFMA.prev_biz_day(date, 0)
    elif convention == 'Modified Preceding':
        adjusted_date = modified_preceding(date)
    else:
        adjusted_date = date
    return adjusted_date


def modified_following(date):
    candidate = SIFMA.next_biz_day(date, 0)
    if candidate.month != date.month:
        return SIFMA.prev_biz_day(date, 0)
    return candidate


def modified_preceding(date):
    candidate = SIFMA.prev_biz_day(date, 0)
    if candidate.month != date.month:
        return SIFMA.next_biz_day(date, 0)
    return candidate


class SOFRSwap:
    def __init__(
            self,
            reference_date=None,
            start_date=None,
            maturity_date=None,
            tenor='5Y',
            notional=1e6,
            coupon=0.05,
            frequency_fixed='12M',  # Annual payments
            frequency_float='12M',  # Annual payments
            day_count='ACT/360',
            business_day_convention='Modified Following',
            roll_convention='None',  # 'None', 'EOM'
            pay_delay=2
    ):
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.tenor = tenor  # Stored for convenience
        self.notional = notional
        self.coupon = coupon
        self.frequency_fixed = frequency_fixed
        self.frequency_float = frequency_float
        self.day_count = day_count
        self.business_day_convention = business_day_convention
        self.roll_convention = roll_convention
        self.pay_delay = pay_delay

        self.calculate_start_end_date(reference_date)
        self.fixed_leg_schedule = self.generate_leg_schedule(self.frequency_fixed)
        self.float_leg_schedule = self.generate_leg_schedule(self.frequency_float)


    def calculate_start_end_date(self, reference_date):
        # Getting the start date right
        if self.start_date is not None:
            try:
                # If start_date is convertible to a date
                self.start_date = pd.Timestamp(self.start_date).to_pydatetime()
            except:
                # If start_date is a tenor string for forward starting swaps
                # Need reference date as input
                reference_date = pd.Timestamp(reference_date).to_pydatetime()
                spot_date = SIFMA.next_biz_day(reference_date, 2)
                num = int(self.start_date[:-1])
                unit = self.start_date[-1].upper()
                if unit == 'Y':
                    self.start_date = spot_date + relativedelta(years=num)
                elif unit == 'M':
                    self.start_date = spot_date + relativedelta(months=num)
                else:
                    raise ValueError("Unsupported tenor unit. Use 'Y' for years or 'M' for months.")
        else:
            reference_date = pd.Timestamp(reference_date).to_pydatetime()
            self.start_date = SIFMA.next_biz_day(reference_date, 2)

        # Getting maturity date right
        try:
            # If maturity_date is given as date
            assert self.maturity_date is not None
            self.maturity_date = pd.Timestamp(self.maturity_date).to_pydatetime()
        except:
            # In this case use tenor and start_date to calculate maturity date
            num = int(self.tenor[:-1])
            unit = self.tenor[-1].upper()
            if unit == 'Y':
                self.maturity_date = self.start_date + relativedelta(years=num)
            elif unit == 'M':
                self.maturity_date = self.start_date + relativedelta(months=num)
            else:
                raise ValueError("Unsupported tenor unit. Use 'Y' for years or 'M' for months.")

        # Roll day Adjust
        if isinstance(self.roll_convention, int):
            try:
                self.maturity_date.replace(day=self.roll_convention)
            except:
                pass

        # EOM roll convention
        if self.roll_convention == "EOM":
            self.maturity_date += MonthEnd(0)
            self.maturity_date = self.maturity_date.to_pydatetime()

    def generate_leg_schedule(self, frequency):
        schedule = []

        freq_num = int(frequency[:-1])
        freq_unit = frequency[-1].upper()

        # Generate unadjusted roll dates
        unadj_date_list = self.generate_unadjusted_dates(self.start_date, self.maturity_date, freq_num, freq_unit)

        # Adjust the roll dates using Modified Following convention by default
        adj_date_list = [adjust_date(d, self.business_day_convention) for d in unadj_date_list]

        # Build the schedule
        for i in range(len(adj_date_list) - 1):
            accrual_start_date = adj_date_list[i]
            accrual_end_date = adj_date_list[i + 1]

            # Calculate payment date with payment delay in SIFMA calendar
            payment_date = SIFMA.next_biz_day(accrual_end_date, self.pay_delay)

            # Calculate day count fraction
            dcf = self.calculate_day_count_fraction(accrual_start_date, accrual_end_date)

            schedule.append({
                'Accrual Start Date': accrual_start_date,
                'Accrual End Date': accrual_end_date,
                'Payment Date': payment_date,
                'Day Count Fraction': dcf
            })

        return pd.DataFrame(schedule)

    def generate_unadjusted_dates(self, start_date, end_date, freq_num, freq_unit):
        # Generate dates from end date backward
        date_list = [end_date]
        current_date = end_date

        while True:
            if freq_unit == 'M':
                prev_date = current_date - relativedelta(months=freq_num)
            elif freq_unit == 'Y':
                prev_date = current_date - relativedelta(years=freq_num)
            else:
                raise ValueError("Unsupported frequency unit. Use 'M' for months or 'Y' for years.")

            # Apply EOM roll convention
            if self.roll_convention == 'EOM':
                prev_date = prev_date + MonthEnd(0)
                prev_date = prev_date.to_pydatetime()

            if prev_date <= start_date:
                if prev_date < start_date:
                    date_list.insert(0, start_date)
                else:
                    date_list.insert(0, prev_date)
                break
            else:
                date_list.insert(0, prev_date)
                current_date = prev_date
        return date_list

    def calculate_day_count_fraction(self, start_date, end_date):
        delta = end_date - start_date
        if self.day_count == 'ACT/360':
            day_count_fraction = delta.days / 360
        else:
            raise ValueError("Unsupported day count convention")
        return day_count_fraction

    def get_fixed_leg_schedule(self):
        return self.fixed_leg_schedule

    def get_float_leg_schedule(self):
        return self.float_leg_schedule

    def get_swap_schedule(self):
        return {
            'Fixed Leg': self.fixed_leg_schedule,
            'Floating Leg': self.float_leg_schedule,
        }


if __name__ == '__main__':
    swap = SOFRSwap(
        reference_date="2024-10-8",
        start_date=None,  # Trade date
        tenor='5Y',
        day_count='ACT/360',  # Both legs use ACT/360
        business_day_convention='Modified Following',
        roll_convention='End of Month',
    )

    fixed_schedule = swap.get_fixed_leg_schedule()
    float_schedule = swap.get_float_leg_schedule()

    print("Fixed Leg Schedule:")
    print(fixed_schedule)

    print("\nFloating Leg Schedule:")
    print(float_schedule)