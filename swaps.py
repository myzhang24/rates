"""
This module generates the swap schedule for a USD SOFR OIS swap with market convention.
Roll convention of End of Month and IMM are supported.
"""

import pandas as pd
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
from holiday import SIFMA


def swap_spot_date(ref_date):
    date = pd.Timestamp(ref_date)
    days_added = 0
    while days_added < 2:
        date += pd.Timedelta(days=1)
        if SIFMA.is_biz_day(date):
            days_added += 1
    return date


class SOFRSwap:
    def __init__(
            self,
            start_date,
            tenor='5Y',
            frequency_fixed='12M',  # Annual payments
            frequency_float='12M',  # Annual payments
            day_count='ACT/360',
            business_day_convention='Modified Following',
            roll_convention='None',  # 'None', 'EOM'
            pay_delay=2,
            coupon=0.05
    ):

        self.start_date = pd.Timestamp(start_date)
        self.tenor = tenor
        self.frequency_fixed = frequency_fixed
        self.frequency_float = frequency_float
        self.day_count = day_count
        self.business_day_convention = business_day_convention
        self.roll_convention = roll_convention
        self.end_date = self.calculate_end_date()
        self.pay_delay = pay_delay

        self.fixed_leg_schedule = self.generate_leg_schedule(self.frequency_fixed)
        self.float_leg_schedule = self.generate_leg_schedule(self.frequency_float)

        self.coupon = coupon
        self.npv = 0.00

    def calculate_end_date(self):
        num = int(self.tenor[:-1])
        unit = self.tenor[-1].upper()
        if unit == 'Y':
            end_date = self.start_date + relativedelta(years=num)
        elif unit == 'M':
            end_date = self.start_date + relativedelta(months=num)
        else:
            raise ValueError("Unsupported tenor unit. Use 'Y' for years or 'M' for months.")

        # Roll day Adjust
        if isinstance(self.roll_convention, int):
            try:
                end_date.replace(day=self.roll_convention)
            except:
                pass

        # EOM roll convention
        if self.roll_convention == "EOM":
            end_date += MonthEnd(0)
        return end_date

    def generate_leg_schedule(self, frequency):
        schedule = []

        freq_num = int(frequency[:-1])
        freq_unit = frequency[-1].upper()

        # Generate unadjusted roll dates
        unadj_date_list = self.generate_unadjusted_dates(self.start_date, self.end_date, freq_num, freq_unit)

        # Adjust the roll dates using Modified Following convention by default
        adj_date_list = [self.adjust_date(d) for d in unadj_date_list]

        # Build the schedule
        for i in range(len(adj_date_list) - 1):
            accrual_start_date = adj_date_list[i]
            accrual_end_date = adj_date_list[i + 1]

            # Calculate payment date with payment delay in SIFMA calendar
            payment_date = self.calculate_payment_date(accrual_end_date)

            # Calculate day count fraction
            dcf = self.calculate_day_count_fraction(accrual_start_date, accrual_end_date)

            schedule.append({
                'Accrual Start Date': accrual_start_date,
                'Accrual End Date': accrual_end_date,
                'Payment Date': payment_date,
                'Day Count Fraction': dcf
            })

        return schedule

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

    def calculate_payment_date(self, base_date):
        payment_date = base_date
        days_added = 0

        while days_added < self.pay_delay:
            payment_date += pd.Timedelta(days=1)
            if SIFMA.is_biz_day(payment_date):
                days_added += 1

        payment_date = self.adjust_date(payment_date)
        return payment_date

    def adjust_date(self, date):
        if self.business_day_convention == 'Following':
            adjusted_date = self.following(date)
        elif self.business_day_convention == 'Modified Following':
            adjusted_date = self.modified_following(date)
        elif self.business_day_convention == 'Preceding':
            adjusted_date = self.preceding(date)
        elif self.business_day_convention == 'Modified Preceding':
            adjusted_date = self.modified_preceding(date)
        else:
            adjusted_date = date
        return adjusted_date

    def following(self, date):
        return SIFMA.next_biz_day(date, 0)

    def modified_following(self, date):
        candidate = SIFMA.next_biz_day(date, 0)
        if candidate.month != date.month:
            return SIFMA.prev_biz_day(date, 0)
        return candidate

    def preceding(self, date):
        return SIFMA.prev_biz_day(date, 0)

    def modified_preceding(self, date):
        candidate = SIFMA.prev_biz_day(date, 0)
        if candidate.month != date.month:
            return SIFMA.next_biz_day(date, 0)
        return candidate

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
    scheduler = SOFRSwap(
        start_date='2024-01-31',  # Trade date
        tenor='5Y',
        day_count='ACT/360',  # Both legs use ACT/360
        business_day_convention='Modified Following',
        roll_convention='End of Month',
    )

    fixed_schedule = scheduler.get_fixed_leg_schedule()
    float_schedule = scheduler.get_float_leg_schedule()

    print("Fixed Leg Schedule:")
    print(f"{'Period':<6} {'Accrual Start':<15} {'Accrual End':<15} {'Payment Date':<15} {'DCF':<10}")
    for i, period in enumerate(fixed_schedule, 1):
        print(
            f"{i:<6} {period['Accrual Start Date'].strftime('%Y-%m-%d'):<15} {period['Accrual End Date'].strftime('%Y-%m-%d'):<15} {period['Payment Date'].strftime('%Y-%m-%d'):<15} {period['Day Count Fraction']:<10.6f}")

    print("\nFloating Leg Schedule:")
    print(f"{'Period':<6} {'Accrual Start':<15} {'Accrual End':<15} {'Payment Date':<15} {'DCF':<10}")
    for i, period in enumerate(float_schedule, 1):
        print(
            f"{i:<6} {period['Accrual Start Date'].strftime('%Y-%m-%d'):<15} {period['Accrual End Date'].strftime('%Y-%m-%d'):<15} {period['Payment Date'].strftime('%Y-%m-%d'):<15} {period['Day Count Fraction']:<10.6f}")
