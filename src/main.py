"""
Calculate energy consumption of specified area in Sundsvall and generate related graphs.

This file has been created for the purpose of calculating energy consumption in a small
area in Sundsvall, Sweden as part of a report for LTU's (Luleå's Technical University)
course W0021T - Mekanik och Elkraftteknik.

References:
Available in references.md

Copyright © 2022 Giuliano Ruggeri
"""
import matplotlib.pyplot as plt

from src.process import Area, House, HouseSize
import src.generate as generate


def gen_all_figures():
    """Generate data and diagrams."""
    ex_small = House(HouseSize.SMALL, temp=20.0)
    ex_medium = House(HouseSize.MEDIUM, temp=20.0)
    ex_large = House(HouseSize.LARGE, temp=20.0)
    area = Area(94, 13, 3)
    generate.fig_daily_avg(area.avg_energy_consumption_by_day)
    generate.fig_monthly_avg_vs_total(
        area.avg_energy_consumption_by_month, area.total_energy_consumption_by_month
    )
    generate.fig_differences_house_size(
        ex_small.energy_consumption_by_month,
        ex_medium.energy_consumption_by_month,
        ex_large.energy_consumption_by_month,
    )
    generate.fig_differences_house_size(
        ex_small.energy_consumption_by_week,
        ex_medium.energy_consumption_by_week,
        ex_large.energy_consumption_by_week,
    )
    generate.fig_differences_house_size(
        ex_small.energy_consumption_by_day,
        ex_medium.energy_consumption_by_day,
        ex_large.energy_consumption_by_day,
    )
    generate.fig_period_by_days(area.avg_energy_consumption_by_day)
    plt.show()


if __name__ == "__main__":
    gen_all_figures()
