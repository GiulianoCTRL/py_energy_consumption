"""
Calculate energy consumption of specified area in Sundsvall and generate related graphs.

This file has been created for the purpose of calculating energy consumption in a small
area in Sundsvall, Sweden as part of a report for LTU's (Luleå's Technical University)
course W0021T - Mekanik och Elkraftteknik.

It contains functions to generate graphs necessary for the report.

References:
Available in references.md

Copyright © 2022 Giuliano Ruggeri
"""
# pylint: disable=no-member
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


# pylint: disable=invalid-name
def fig_daily_avg(data: npt.NDArray[np.float32]):
    """Generate figure with daily averages."""
    _, ax = plt.subplots()
    x = np.arange(365, dtype=np.int32)
    ax.set_title("Energiförbrukning per dygn")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Dygn")

    sizes = data / (np.ones(shape=(len(data))) * 10) * np.random.uniform(0.95, 1.05)
    colors = np.random.uniform(15, 80, len(data))
    ax.scatter(x, data, sizes=sizes, c=colors)
    ax.plot(x, data, linewidth=1, color="blue", label="medelvärde")


# pylint: disable=invalid-name
def fig_monthly_avg_vs_total(
    data_avg: npt.NDArray[np.float32], data_tot: npt.NDArray[np.float32]
):
    """Generate monthly average vs total."""

    x = np.arange(12, dtype=np.int32)
    _, ax = plt.subplots()

    ax.set_title("Elförbrukning total och medelvärde per månad")
    ax.set_xlabel("Månad")
    ax.set_ylabel("kWh")

    ax.fill_between(x, data_avg, data_tot, alpha=0.5, linewidth=0)
    ax.set_yscale("log")
    ax.plot(x, data_tot, linewidth=2, color="red", label="total")
    ax.plot(x, data_avg, linewidth=2, color="blue", label="medelvärde")
    ax.plot(x, data_tot / data_avg, linewidth=2, color="green", label="kvotient")
    ax.legend()


# pylint: disable=invalid-name
def fig_period_by_days(data: npt.NDArray[np.float32]):
    """Generate daily consumption averaged weekly and monthly."""
    weeks_by_days = np.empty(shape=(52, 7))
    months_by_days = np.empty(shape=(12, 30))
    # Get a 52 x 7 matrix
    for index, start_day in enumerate(range(0, 364, 7)):
        weeks_by_days[index] = data[start_day : start_day + 7]

    # Get a 12 x 30 matrix
    for index, start_day in enumerate(range(0, 360, 30)):
        months_by_days[index] = data[start_day : start_day + 30]

    fig_weeks, ax_weeks = plt.subplots()
    ax_weeks.set_xlabel("dygn")
    ax_weeks.set_ylabel("vecka")
    ax_weeks.set_title("Dagsförbrukning alla veckor")

    im_weeks = ax_weeks.pcolormesh(weeks_by_days)
    fig_weeks.colorbar(im_weeks, ax=ax_weeks, label="kWh")
    ax_weeks.legend()

    fig_months, ax_months = plt.subplots()
    ax_months.set_xlabel("dygn")
    ax_months.set_ylabel("månad")
    ax_months.set_title("Dagsförbrukning alla månader")

    im = ax_months.pcolormesh(months_by_days)
    fig_months.colorbar(im, ax=ax_months, label="kWh")
    ax_months.legend()


# pylint: disable=invalid-name
def fig_differences_house_size(
    data_small: npt.NDArray[np.float32],
    data_medium: npt.NDArray[np.float32],
    data_large: npt.NDArray[np.float32],
):
    """Generate graph with differences in house size."""
    time_period = ""
    if (length := len(data_small)) == 12:
        time_period = "månad"
    elif length == 52:
        time_period = "vecka"
    elif length == 365:
        time_period = "dygn"

    x = np.arange(len(data_small), dtype=np.int32)
    _, ax = plt.subplots()

    ax.set_title(
        f"Elförbrukning - skillnader i hus storlek (Samma temperatur) per {time_period}"
    )
    ax.set_xlabel(time_period)
    ax.set_ylabel("kWh")

    ax.fill_between(x, data_small, data_large, alpha=0.5, linewidth=0)
    ax.plot(x, data_small, linewidth=2, color="blue", label="små hus")
    ax.plot(x, data_medium, linewidth=2, color="green", label="medium hus")
    ax.plot(x, data_large, linewidth=2, color="red", label="stor hus")
