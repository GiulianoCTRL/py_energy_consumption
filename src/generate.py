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

plt.style.use("dark_background")


def fig_daily_avg(data: npt.NDArray[np.float32], save: bool = False, prefix: str = ""):
    """Generate figure with daily averages."""
    fig, ax = plt.subplots()
    x = np.arange(1, 366, dtype=np.int32)
    ax.set_title(prefix + "Energiförbrukning per dygn (medelvärde)")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Dygn")

    # Just giving the data points different sizes and colours for visibility
    sizes = data / (np.ones(shape=(len(data))) * 10) * np.random.uniform(0.95, 1.05)
    colors = np.random.uniform(15, 80, len(data))
    ax.scatter(x, data, sizes=sizes, c=colors)
    ax.plot(x, data, linewidth=1, color="blue", label="medelvärde")
    if save:
        fig.savefig("graphs/" + prefix + "01_Daily_average.png", dpi=500)


def fig_monthly_total(
    data: npt.NDArray[np.float32], save: bool = False, prefix: str = ""
):
    """Generate monthly total."""

    x = np.arange(12, dtype=np.int32)
    fig, ax = plt.subplots()

    ax.set_title(prefix + "Elförbrukning total per månad")
    ax.set_xlabel("Månad")
    ax.set_ylabel("kWh")

    ax.plot(x, data, linewidth=2, color="red")
    if save:
        fig.savefig("graphs/" + prefix + "02_Total_monthly.png", dpi=500)


def fig_monthly_avg(data: npt.NDArray[np.float32], save: bool = False, prefix: str = ""):
    """Generate monthly average."""

    x = np.arange(12, dtype=np.int32)
    fig, ax = plt.subplots()

    ax.set_title(prefix + "Elförbrukning avg per månad")
    ax.set_xlabel("Månad")
    ax.set_ylabel("kWh")

    ax.plot(x, data, linewidth=2, color="blue")
    if save:
        fig.savefig("graphs/" + prefix + "03_monthly_average.png", dpi=500)


def fig_days_over_interval(
    steps: int, data: npt.NDArray[np.float32], save: bool = False, prefix: str = ""
):
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
    ax.set_title(prefix + f"Förbrukning dag i {label[0]}")

    im = ax.pcolormesh(interval_by_days)
    fig.colorbar(im, ax=ax, label="kWh")
    if save:
        fig.savefig("graphs/" + prefix + f"{label[2]}_daily_over_{label[1]}.png", dpi=500)


def fig_compare_interval_by_week(
    data: npt.NDArray[np.float32], save: bool = False, prefix: str = ""
):
    """Generate comparison of each interval by week."""
    fig, ax = plt.subplots()
    ax.set_ylabel("kWh")
    ax.set_xlabel("vecka")
    ax.set_title(prefix + "Förbrukning för alla veckor (total)")
    ax.bar(range(1, 53), data, width=1, edgecolor="white", linewidth=1, color="blue")
    ax.set(xticks=range(4, 53, 4))
    if save:
        fig.savefig("graphs/" + prefix + "06_total_by_week.png", dpi=500)


def fig_compare_interval_by_month(
    data: npt.NDArray[np.float32], save: bool = False, prefix: str = ""
):
    """Generate comparison of each interval by it's steps."""
    fig, ax = plt.subplots()
    ax.set_ylabel("kWh")
    ax.set_xlabel("månad")
    ax.set_title(prefix + "Förbrukning för alla månader (total)")
    ax.bar(range(1, 13), data, edgecolor="white", width=1, linewidth=1, color="blue")
    ax.set(xticks=range(3, 13, 3))
    if save:
        fig.savefig("graphs/" + prefix + "07_total_by_month.png", dpi=500)


def fig_temp_diff_stackplot(
    data: npt.NDArray[np.float32], save: bool = False, prefix: str = ""
):
    """Generate stackplot comparing temperature differences."""
    x = np.arange(len(data[0]))

    fig, ax = plt.subplots()
    ax.set_xlabel("månad")
    ax.set_ylabel("kWh")
    ax.set_title(prefix + "Månadsförbrukning för olika temperaturer")
    ax.stackplot(x, data, labels=[str(i) + " °C" for i in range(20, 25)])
    ax.legend(loc="lower right")
    if save:
        fig.savefig("graphs/" + prefix + "08_monthly_different_temps.png", dpi=500)


def fig_differences_house_size(
    data_small: npt.NDArray[np.float32],
    data_medium: npt.NDArray[np.float32],
    data_large: npt.NDArray[np.float32],
    save: bool = False,
    prefix: str = "",
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

    ax.set_title(prefix + f"Skillnader i förbrukning(Samma temp) per {time_period[0]}")
    ax.set_xlabel(time_period[0])
    ax.set_ylabel("kWh")

    ax.fill_between(x, data_small, data_large, alpha=0.5, linewidth=0)
    ax.plot(x, data_small, linewidth=2, color="blue", label="små hus")
    ax.plot(x, data_medium, linewidth=2, color="green", label="medium hus")
    ax.plot(x, data_large, linewidth=2, color="red", label="stor hus")
    ax.legend(loc="center right")
    if save:
        fig.savefig(
            "graphs/"
            + prefix
            + f"{time_period[2]}_different_house_consumption_by_{time_period[1]}.png",
            dpi=500,
        )


def fig_daily_total(data: npt.NDArray[np.float32], save: bool = False, prefix: str = ""):
    """Generate figure with daily total."""
    fig, ax = plt.subplots()
    x = np.arange(1, 366, dtype=np.int32)
    ax.set_title(prefix + "Energiförbrukning per dygn (total)")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Dygn")

    # Just giving the data points different sizes and colours for visibility
    sizes = data / (np.ones(shape=(len(data))) * 2000) * np.random.uniform(0.95, 1.05)
    colors = np.random.uniform(15, 80, len(data))
    ax.scatter(x, data, sizes=sizes, c=colors)
    ax.plot(x, data, linewidth=1, color="blue", label="total")
    if save:
        fig.savefig("graphs/" + prefix + "12_Daily_total.png", dpi=500)


def fig_stack_all_but_sawmill_by_interval(
    data: npt.NDArray[np.float32],
    save: bool = False,
    prefix: str = "",
):
    """
    Compare values of house and street light consumption vs solar power production.

    :param data:        [solarpanel, properties, vehicles, lights]
    """
    fig, ax = plt.subplots()
    interval = (366, "Daglig", "Dygn", "daily", 13)
    if len(data[0]) == 12:
        interval = (13, "Månadsvis", "Månad", "monthly", 14)
    elif len(data[0]) == 52:
        interval = (53, "Veckovis", "vecka", "weekly", 15)
    x = np.arange(1, interval[0], dtype=np.int32)
    ax.set_title(prefix + f"{interval[1]} Solarkraft & Hushåll & Elbil & Gatubelysning.")
    ax.set_ylabel("kWh")
    ax.set_xlabel(interval[2])

    ax.stackplot(x, data[1:4], labels=["Hushåll", "Elbil", "Gatubelysning"])
    ax.plot(x, data[0], linewidth=3, color="red", label="solarkraft (genererad)")
    ax.legend()
    if save:
        fig.savefig(
            "graphs/" + prefix + f"{interval[4]}_Households_street_solar_{interval[3]}.png", dpi=500
        )


def fig_compare_all_data(
    data: npt.NDArray[np.float32], save: bool = False, prefix: str = ""
):
    """
    Compare all available data.

    :param data:        [properties, vechiles, lights, swamill, solar]
    """
    length = len(data[0])
    x = np.arange(length)
    label = ""
    if length == 365:
        label = "Dag"
    elif length == 52:
        label = "Vecka"
    elif length == 12:
        label = "Månad"
    fig, ax_kwh = plt.subplots()
    ax_kwh.set_xlabel(label)
    ax_kwh.set_ylabel("kWh")
    ax_kwh.set_title(prefix + f"{label}sförbrukning och generation alla")
    ax_kwh.plot(x, data[0], linewidth=2, label="Fastigheter (kWh)", color="green")
    ax_kwh.plot(x, data[1], linewidth=2, label="Elbil (kWh)", color="blue")
    ax_kwh.plot(x, data[2], linewidth=2, label="Gatubelysning (kWh)", color="orange")
    ax_kwh.plot(x, data[4], linewidth=2, label="Solarkraft (genererad) (kWh)", color="purple")
    ax_mwh = ax_kwh.twinx()
    ax_mwh.set_ylabel("MWh")
    ax_mwh.plot(x, data[3] / 1000, linewidth=2, label="Sågverk (MWh)", color="red")
    ax_mwh.plot(
        x,
        (data[0] + data[1] + data[2] + data[3] - data[4]) / 1000,
        linewidth=2,
        label="Total (MWh)",
        color="white",
    )
    ax_kwh.legend()
    ax_mwh.legend(loc="lower right")
    if save:
        fig.savefig("graphs/" + prefix + "16_All_data_compared.png", dpi=500)


def fig_pie_chart_yearly_cons(data: npt.NDArray, save: bool = False, prefix: str = ""):
    """
    Show pie chart of consumption by percentage.

    :param data:        [properties, vehicles, lights]
    """
    fig, ax = plt.subplots()
    colors = colors = plt.get_cmap("Blues")(np.linspace(0.2, 0.7, len(data)))
    ax.pie(data, colors=colors, labels=["Hushåll", "Elbil", "Gatubelysning"])
    ax.set_title(prefix + "Fördelning av årsförbrukning (privat och statlig)")
    ax.legend()
    if save:
        fig.savefig("graphs/" + prefix + "17_Consumers_pie_chart.png", dpi=500)


def fig_plot_compare_now_with_future(
    data: npt.NDArray, save: bool = False, prefix: str = ""
):
    """
    Compare area data now and in the future.

    :param data:    [data_now, data_5years, data_10years]
    """
    fig, ax = plt.subplots()
    x = np.arange(len(data[0]))
    ax.set_title(prefix + "Energiförbrukning i privata sektorn per vecka jämfört med framtiden")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Vecka")

    ax.plot(x, data[0], linewidth=2, color="green", label="2022")
    ax.plot(x, data[1], linewidth=2, color="orange", label="2027")
    ax.plot(x, data[2], linewidth=2, color="blue", label="2032")
    ax.legend()
    if save:
        fig.savefig("graphs/" + prefix + "19_Comparison_future.png", dpi=500)
