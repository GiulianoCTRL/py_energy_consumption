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


def fig_daily_avg(data: npt.NDArray[np.float32]):
    """Generate figure with daily averages."""
    _fig, ax = plt.subplots()
    x = np.arange(1, 366, dtype=np.int32)
    ax.set_title("Energiförbrukning per dygn")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Dygn")

    sizes = data / (np.ones(shape=(len(data))) * 10) * np.random.uniform(0.95, 1.05)
    colors = np.random.uniform(15, 80, len(data))
    ax.scatter(x, data, sizes=sizes, c=colors)
    ax.plot(x, data, linewidth=1, color="blue", label="medelvärde")


def fig_monthly_total(data: npt.NDArray[np.float32]):
    """Generate monthly total."""

    x = np.arange(12, dtype=np.int32)
    _fig, ax = plt.subplots()

    ax.set_title("Elförbrukning total per månad")
    ax.set_xlabel("Månad")
    ax.set_ylabel("kWh")

    ax.plot(x, data, linewidth=2, color="red")


def fig_monthly_avg(data: npt.NDArray[np.float32]):
    """Generate monthly average."""

    x = np.arange(12, dtype=np.int32)
    _fig, ax = plt.subplots()

    ax.set_title("Elförbrukning avg per månad")
    ax.set_xlabel("Månad")
    ax.set_ylabel("kWh")

    ax.plot(x, data, linewidth=2, color="blue")


def fig_days_over_interval(steps: int, data: npt.NDArray[np.float32]):
    """Generate average daily consumption over interval."""
    days = 365 // steps  # matrix rows
    interval_by_days = np.empty(shape=(steps, days))  # matrix columns

    # Fill a row x column matrix
    for index, start_day in enumerate(range(0, days * steps, days)):
        interval_by_days[index] = data[start_day: start_day + days]

    fig, ax = plt.subplots()
    ax.set_xlabel("dygn")
    label = "månad" if days == 30 else "vecka"
    ax.set_ylabel(label)
    ax.set_title(f"Förbrukning dag i {label}")

    im = ax.pcolormesh(interval_by_days)
    fig.colorbar(im, ax=ax, label="kWh")


def fig_compare_interval_by_week(data: npt.NDArray[np.float32]):
    """Generate comparison of each interval by it's steps."""
    _fig, ax = plt.subplots()
    ax.set_ylabel("kWh")
    ax.set_xlabel("vecka")
    ax.set_title("Förbrukning för alla veckor (total)")
    ax.bar(range(1, 53), data, width=1, edgecolor="white", linewidth=1)
    ax.set(xticks=range(4, 53, 4))


def fig_compare_interval_by_month(data: npt.NDArray[np.float32]):
    """Generate comparison of each interval by it's steps."""
    _fig, ax = plt.subplots()
    ax.set_ylabel("kWh")
    ax.set_xlabel("månad")
    ax.set_title("Förbrukning för alla månader (total)")
    ax.bar(range(1, 13), data, edgecolor="white", width=1, linewidth=1)
    ax.set(xticks=range(3, 13, 3))


def fig_temp_diff_stackplot(data: npt.NDArray[np.float32]):
    """Generate stackplot comparing temperature differences."""
    x = np.arange(len(data[0]))
    y = np.vstack(data)

    _fig, ax = plt.subplots()
    ax.set_xlabel("månad")
    ax.set_ylabel("kWh")
    ax.set_title("Månadsförbrukning för olika temperaturer (fjärrvärme)")
    ax.stackplot(x, y, labels=[str(i) + " °C" for i in range(20, 25)])
    ax.legend(loc="lower right")


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
    _fig, ax = plt.subplots()

    ax.set_title(f"Skillnader i förbrukning(Samma temp, fjärrvärme) per {time_period}")
    ax.set_xlabel(time_period)
    ax.set_ylabel("kWh")

    ax.fill_between(x, data_small, data_large, alpha=0.5, linewidth=0)
    ax.plot(x, data_small, linewidth=2, color="blue", label="små hus")
    ax.plot(x, data_medium, linewidth=2, color="green", label="medium hus")
    ax.plot(x, data_large, linewidth=2, color="red", label="stor hus")
    ax.legend(loc="center right")
