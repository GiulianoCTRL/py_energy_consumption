"""
Calculate energy consumption of specified area in Sundsvall and generate related graphs.

This file has been created for the purpose of calculating energy consumption in a small
area in Sundsvall, Sweden as part of a report for LTU's (Luleå's Technical University)
course W0021T - Mekanik och Elkraftteknik.

It contains functions to generate graphs necessary for the report.

References:
Available in references.md
"""
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def fig_daily_avg(data: npt.NDArray[np.float32], save: bool = False):
    """Generate figure with daily averages."""
    fig, ax = plt.subplots()
    x = np.arange(1, 366, dtype=np.int32)
    ax.set_title("Energiförbrukning per dygn")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Dygn")

    sizes = data / (np.ones(shape=(len(data))) * 10) * np.random.uniform(0.95, 1.05)
    colors = np.random.uniform(15, 80, len(data))
    ax.scatter(x, data, sizes=sizes, c=colors)
    ax.plot(x, data, linewidth=1, color="blue", label="medelvärde")
    if save:
        fig.savefig("01Daily_average.png", dpi=500)


def fig_monthly_total(data: npt.NDArray[np.float32], save: bool = False):
    """Generate monthly total."""

    x = np.arange(12, dtype=np.int32)
    fig, ax = plt.subplots()

    ax.set_title("Elförbrukning total per månad")
    ax.set_xlabel("Månad")
    ax.set_ylabel("kWh")

    ax.plot(x, data, linewidth=2, color="red")
    if save:
        fig.savefig("02Total_monthly.png", dpi=500)


def fig_monthly_avg(data: npt.NDArray[np.float32], save: bool = False):
    """Generate monthly average."""

    x = np.arange(12, dtype=np.int32)
    fig, ax = plt.subplots()

    ax.set_title("Elförbrukning avg per månad")
    ax.set_xlabel("Månad")
    ax.set_ylabel("kWh")

    ax.plot(x, data, linewidth=2, color="blue")
    if save:
        fig.savefig("03monthly_average.png", dpi=500)


def fig_days_over_interval(steps: int, data: npt.NDArray[np.float32], save: bool = False):
    """Generate average daily consumption over interval."""
    days = 365 // steps  # matrix rows
    interval_by_days = np.empty(shape=(steps, days))  # matrix columns

    # Fill a row x column matrix
    for index, start_day in enumerate(range(0, days * steps, days)):
        interval_by_days[index] = data[start_day : start_day + days]

    fig, ax = plt.subplots()
    ax.set_xlabel("dygn")
    label = ("månad", "month", "04") if days == 30 else ("vecka", "week", "05")
    ax.set_ylabel(label[0])
    ax.set_title(f"Förbrukning dag i {label[0]}")

    im = ax.pcolormesh(interval_by_days)
    fig.colorbar(im, ax=ax, label="kWh")
    if save:
        fig.savefig(f"{label[2]}daily_over_{label[1]}.png", dpi=500)


def fig_compare_interval_by_week(data: npt.NDArray[np.float32], save: bool = False):
    """Generate comparison of each interval by week."""
    fig, ax = plt.subplots()
    ax.set_ylabel("kWh")
    ax.set_xlabel("vecka")
    ax.set_title("Förbrukning för alla veckor (total)")
    ax.bar(range(1, 53), data, width=1, edgecolor="white", linewidth=1)
    ax.set(xticks=range(4, 53, 4))
    if save:
        fig.savefig("06total_by_week.png", dpi=500)


def fig_compare_interval_by_month(data: npt.NDArray[np.float32], save: bool = False):
    """Generate comparison of each interval by it's steps."""
    fig, ax = plt.subplots()
    ax.set_ylabel("kWh")
    ax.set_xlabel("månad")
    ax.set_title("Förbrukning för alla månader (total)")
    ax.bar(range(1, 13), data, edgecolor="white", width=1, linewidth=1)
    ax.set(xticks=range(3, 13, 3))
    if save:
        fig.savefig("07total_by_month.png", dpi=500)


def fig_temp_diff_stackplot(data: npt.NDArray[np.float32], save: bool = False):
    """Generate stackplot comparing temperature differences."""
    x = np.arange(len(data[0]))

    fig, ax = plt.subplots()
    ax.set_xlabel("månad")
    ax.set_ylabel("kWh")
    ax.set_title("Månadsförbrukning för olika temperaturer")
    ax.stackplot(x, data, labels=[str(i) + " °C" for i in range(20, 25)])
    ax.legend(loc="lower right")
    if save:
        fig.savefig("08monthly_different_temps.png", dpi=500)


def fig_differences_house_size(
    data_small: npt.NDArray[np.float32],
    data_medium: npt.NDArray[np.float32],
    data_large: npt.NDArray[np.float32],
    save: bool = False,
):
    """Generate graph with differences in house size."""
    time_period = ("", "", "")
    if (length := len(data_small)) == 12:
        time_period = ("månad", "month", "09")
    elif length == 52:
        time_period = ("vecka", "week", "10")
    elif length == 365:
        time_period = ("dygn", "day", "11")

    x = np.arange(len(data_small), dtype=np.int32)
    fig, ax = plt.subplots()

    ax.set_title(f"Skillnader i förbrukning(Samma temp) per {time_period[0]}")
    ax.set_xlabel(time_period[0])
    ax.set_ylabel("kWh")

    ax.fill_between(x, data_small, data_large, alpha=0.5, linewidth=0)
    ax.plot(x, data_small, linewidth=2, color="blue", label="små hus")
    ax.plot(x, data_medium, linewidth=2, color="green", label="medium hus")
    ax.plot(x, data_large, linewidth=2, color="red", label="stor hus")
    ax.legend(loc="center right")
    if save:
        fig.savefig(
            f"{time_period[2]}different_house_consumption_by_{time_period[1]}.png",
            dpi=500,
        )
