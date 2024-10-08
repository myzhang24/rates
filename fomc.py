import datetime as dt


def generate_fomc_meeting_dates(start_date: dt.datetime, end_date: dt.datetime):
    """
    Generates FOMC meeting dates within the given date range.

    Args:
        start_date (datetime.date): The start date of the range.
        end_date (datetime.date): The end date of the range.

    Returns:
        meetings (list of date): List of FOMC meeting dates (second day of each meeting).
    """
    # Actual FOMC meeting dates from 2019 to 2026
    actual_meetings = [
        # 2019
        dt.datetime(2019, 1, 30),
        dt.datetime(2019, 3, 20),
        dt.datetime(2019, 5, 1),
        dt.datetime(2019, 6, 19),
        dt.datetime(2019, 7, 31),
        dt.datetime(2019, 9, 18),
        dt.datetime(2019, 10, 30),
        dt.datetime(2019, 12, 11),
        # 2020
        dt.datetime(2020, 1, 29),
        dt.datetime(2020, 3, 18),
        dt.datetime(2020, 4, 29),
        dt.datetime(2020, 6, 10),
        dt.datetime(2020, 7, 29),
        dt.datetime(2020, 9, 16),
        dt.datetime(2020, 11, 5),
        dt.datetime(2020, 12, 16),
        # 2021
        dt.datetime(2021, 1, 27),
        dt.datetime(2021, 3, 17),
        dt.datetime(2021, 4, 28),
        dt.datetime(2021, 6, 16),
        dt.datetime(2021, 7, 28),
        dt.datetime(2021, 9, 22),
        dt.datetime(2021, 11, 3),
        dt.datetime(2021, 12, 15),
        # 2022
        dt.datetime(2022, 1, 26),
        dt.datetime(2022, 3, 16),
        dt.datetime(2022, 5, 4),
        dt.datetime(2022, 6, 15),
        dt.datetime(2022, 7, 27),
        dt.datetime(2022, 9, 21),
        dt.datetime(2022, 11, 2),
        dt.datetime(2022, 12, 14),
        # 2023
        dt.datetime(2023, 2, 1),
        dt.datetime(2023, 3, 22),
        dt.datetime(2023, 5, 3),
        dt.datetime(2023, 6, 14),
        dt.datetime(2023, 7, 26),
        dt.datetime(2023, 9, 20),
        dt.datetime(2023, 11, 1),
        dt.datetime(2023, 12, 13),
        # 2024
        dt.datetime(2024, 1, 31),
        dt.datetime(2024, 3, 20),
        dt.datetime(2024, 5, 1),
        dt.datetime(2024, 6, 12),
        dt.datetime(2024, 7, 31),
        dt.datetime(2024, 9, 18),
        dt.datetime(2024, 11, 7),
        dt.datetime(2024, 12, 18),
        # 2025
        dt.datetime(2025, 1, 29),
        dt.datetime(2025, 3, 19),
        dt.datetime(2025, 5, 7),
        dt.datetime(2025, 6, 18),
        dt.datetime(2025, 7, 30),
        dt.datetime(2025, 9, 17),
        dt.datetime(2025, 10, 29),
        dt.datetime(2025, 12, 10),
        # 2026
        dt.datetime(2026, 1, 28),
        dt.datetime(2026, 3, 18),
        dt.datetime(2026, 4, 29),
        dt.datetime(2026, 6, 17),
        dt.datetime(2026, 7, 29),
        dt.datetime(2026, 9, 16),
        dt.datetime(2026, 10, 28),
        dt.datetime(2026, 12, 9),
    ]

    meetings = []

    # Add actual meetings within the date range
    for meeting_date in actual_meetings:
        if start_date <= meeting_date <= end_date:
            meetings.append(meeting_date)

    # Generate estimated meetings outside the range of actual dates
    if start_date.year < 2019 or end_date.year > 2026:
        estimated_meetings = estimate_future_meeting_dates(start_date, end_date)
        meetings.extend(estimated_meetings)

    # Remove duplicates and sort the list
    meetings = sorted(list(set(meetings)))

    return meetings


def estimate_future_meeting_dates(start_date, end_date):
    """
    Estimates FOMC meeting dates using heuristics for years outside 2019-2026.

    Args:
        start_date (datetime.date): The start date of the range.
        end_date (datetime.date): The end date of the range.

    Returns:
        estimated_meetings (list of date): List of estimated meeting dates.
    """
    estimated_meetings = []

    # Determine years that need estimation
    years = set()
    current_year = start_date.year
    while current_year <= end_date.year:
        if current_year < 2019 or current_year > 2026:
            years.add(current_year)
        current_year += 1

    for year in years:
        # Estimate meetings for the year
        meetings = estimate_meetings_for_year(year)
        for meeting_date in meetings:
            if start_date <= meeting_date <= end_date:
                estimated_meetings.append(meeting_date)

    return estimated_meetings

def estimate_meetings_for_year(year):
    """
    Estimate the FOMC meeting dates for a given year.

    Parameters:
        year (int): The year for which to estimate meeting dates.

    Returns:
        List[datetime.datetime]: A list of 8 datetime objects representing the estimated meeting dates.
    """
    meetings = []
    # Approximate target days for each meeting based on historical patterns
    meeting_info = [
        (1, 25),   # Late January
        (3, 15),   # Mid-March
        (5, 3),    # Early May
        (6, 14),   # Mid-June
        (7, 26),   # Late July
        (9, 20),   # Mid-September
        (11, 1),   # Early November
        (12, 13),  # Mid-December
    ]
    for month, day in meeting_info:
        # Create the target date
        target_date = dt.datetime(year, month, day)
        # Calculate days to the next Wednesday (weekday 2)
        days_ahead = (2 - target_date.weekday()) % 7
        # Adjust the date to the next Wednesday
        meeting_date = target_date + dt.timedelta(days=days_ahead)
        # Combine date with a default time (midnight)
        meetings.append(meeting_date)
    return meetings


# Example usage:
if __name__ == "__main__":
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime(2030, 12, 31)
    fomc_meetings = generate_fomc_meeting_dates(start, end)
    print("FOMC Meeting Dates (Second Day) between {} and {}:".format(start, end))
    for meeting in fomc_meetings:
        print(meeting.strftime("%B %d, %Y"))
