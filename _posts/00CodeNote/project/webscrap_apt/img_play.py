import logging
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Logs will go to CloudWatch log group corresponding to lambda,
# If Lambda has the necessary IAM permissions.
# Set logLevel to logging.INFO or logging.DEBUG for debugging.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOGGER = logging.getLogger(__name__)
# Retrieve log level from Lambda Environment Variables
LOGGER.setLevel(level=os.environ.get("LOG_LEVEL", "INFO").upper())


def read_csv(dir_path):
    # list to store files
    csv_files = []
    # csv_files = ["./apt_output/apt_20230326.csv", "./apt_output/apt_20230327.csv", "./apt_output/apt_20230328.csv"]
    # Iterate directory, get list of CSV files to read
    dir_path = dir_path + "apt_output/"
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            dir_path + path
            csv_files.append(dir_path + path)
            # LOGGER.info("Add file: %s " % file_name)
    LOGGER.info("🍰 ======= Collected data in csv_files: %s =======" % csv_files)
    return csv_files


def floor_plan_list(csv_files):
    # Create a dictionary with Apt names as keys and a list of unique Floor_plan values as values
    apt_dict = {}
    # loop through each CSV file and append to all_data
    for file in csv_files:
        df = pd.read_csv(file)
        # Remove duplicates
        df = df.drop_duplicates(subset=["Apt", "Floor_plan"])
        for apt in df["Apt"].unique():
            floor_plan_list = df[df["Apt"] == apt]["Floor_plan"].unique().tolist()
            apt_dict[apt] = floor_plan_list
    LOGGER.info("🍰 ======= Collected floorplan in Apts =======")
    for apt in apt_dict:
        LOGGER.info("%s: %s", apt, apt_dict[apt])
    return apt_dict


def draw_png(csv_files, apt, floor_plans):
    # Create a new figure to plot the data
    fig, ax = plt.subplots()
    # create empty dataframe to store all data
    all_data = pd.DataFrame()

    # loop through each CSV file and append to all_data
    for file in csv_files:
        LOGGER.info(f"Data info appending for file {file}")
        df = pd.read_csv(file, parse_dates=["Date"])
        # all_data = all_data.append(df)
        all_data = pd.concat([all_data, df])

    # filter data by floor plans
    filtered_data = all_data[all_data["Floor_plan"].isin(floor_plans)]

    # loop through each floor plan and plot on the same graph
    for floor_plan in floor_plans:
        data = filtered_data[filtered_data["Floor_plan"] == floor_plan]
        data = data.copy()
        # modify the format and sort
        data["Rent"] = (
            data["Rent"].astype(str).str.replace(",", "").str.extract(r"\$(.*)/month")
        )
        # sort the data by date
        data = data.sort_values(by="Date")
        ax.plot(data["Date"], data["Rent"].astype(float), label=floor_plan)

        # add marker for lowest rent price
        min_rent = data["Rent"].astype(float).min()
        # LOGGER.info("Minimum rent for floor plan %s is %s ", floor_plan, min_rent)
        # print(type(min_rent))
        if np.isnan(min_rent):
            LOGGER.info("-- No data for floor plan %s.", floor_plan)
        else:
            data["Rent"] = data["Rent"].astype(float)
            min_rent = data[data["Rent"] == min_rent]["Rent"].iloc[0]
            min_date = data[data["Rent"] == min_rent]["Date"].iloc[0]
            # convert the date format to "YYYY/MM/DD"
            min_date_str = min_date.strftime("%Y/%m/%d")
            LOGGER.info(
                "-- Minimum rent for floor plan %s is %s on %s",
                floor_plan,
                min_rent,
                min_date_str,
            )

            ax.annotate(
                f"${min_rent} ({min_date_str})",
                xy=(min_date, min_rent),
                xytext=(min_date, min_rent + 100),
                ha="left",
                va="center",
                fontsize=4,
                color="blue",
                arrowprops=dict(arrowstyle="->", color="blue"),
            )
            ax.plot(min_date, min_rent, marker="o", markersize=3, color="green")

    # set axis labels and legend
    plt.xlabel("Date")
    plt.ylabel("Rent ($)")

    # plt.legend()
    ax.legend(title=f"{apt}-Floor_plan", bbox_to_anchor=[0.5, 0.5], loc="center right")

    # # show the plot
    # plt.show()

    # Save the plot as a PNG file
    plt.savefig(f"{DIR_PATH}/APT-{apt}.png", dpi=300)

    # Clear the plot for the next CSV file
    ax.clear()
    LOGGER.info("======= update apt png for APT %s =======" % apt)


def main(dir_path):
    csv_files = read_csv(dir_path)
    apt_dict = floor_plan_list(csv_files)
    for apt, floor_plans in apt_dict.items():
        LOGGER.info("🍰 ======= draw apt png for APT %s =======" % apt)
        draw_png(csv_files, apt, floor_plans)


if __name__ == "__main__":
    # DIR_PATH = './apt_output/'
    DIR_PATH = "./_posts/00CodeNote/project/webscrap_apt/"
    LOGGER.info("🍰 The DIR_PATH has been set to: %s", DIR_PATH)
    main(DIR_PATH)
