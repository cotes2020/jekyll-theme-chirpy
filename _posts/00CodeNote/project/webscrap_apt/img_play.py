import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker


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
    dir_path = dir_path + 'apt_output/'
    for path in os.listdir(dir_path): 
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            csv_files.append(dir_path+path)
    LOGGER.info("\n======= Collected data in csv_files: %s =======" % csv_files)
    return csv_files


def floow_plan_list(csv_files):
    # Create a dictionary with Apt names as keys and a list of unique Floor_plan values as values
    apt_dict = {}
    # loop through each CSV file and append to all_data
    for file in csv_files:
        df = pd.read_csv(file)  
        # Remove duplicates
        df = df.drop_duplicates(subset=['Apt', 'Floor_plan'])
        for apt in df['Apt'].unique():
            floor_plan_list = df[df['Apt'] == apt]['Floor_plan'].unique().tolist()
            apt_dict[apt] = floor_plan_list
    return apt_dict


def draw_png(csv_files, apt, floor_plans):
    # Create a new figure to plot the data
    fig, ax = plt.subplots()

    # create empty dataframe to store all data
    all_data = pd.DataFrame()

    # loop through each CSV file and append to all_data
    for file in csv_files:
        df = pd.read_csv(file, parse_dates=["Date"]) 
        # all_data = all_data.append(df)
        # outputxlsx = pd.concat([all_data, df])
        all_data = pd.concat([all_data, df])
        
    # print(outputxlsx)

    # filter data by floor plans
    filtered_data = all_data[all_data["Floor_plan"].isin(floor_plans)]

    # loop through each floor plan and plot on the same graph
    for floor_plan in floor_plans:
        data = filtered_data[filtered_data["Floor_plan"] == floor_plan] 
        data = data.copy()
        # modify the formate and sort
        data["Rent"] = data["Rent"].astype(str).str.replace(',', '').str.extract(r'\$(.*)/month')
        data = data.sort_values(by="Date")  # sort the data by date
        # print(data["Rent"])
        ax.plot(data["Date"], data["Rent"].astype(float), label=floor_plan)

    # set axis labels and legend
    plt.xlabel("Date")
    plt.ylabel("Rent ($)")

    # plt.legend() 
    ax.legend(title=f"{apt}-Floor_plan", bbox_to_anchor=[0.5, 0.5], loc='center right')

    # # show the plot
    # plt.show()

    # Save the plot as a PNG file
    plt.savefig(f'{DIR_PATH}/APT-{apt}.png')
    
    # Clear the plot for the next CSV file
    ax.clear()
    LOGGER.info("\n======= update apt png for APT %s =======" % apt)
    
    
def main(dir_path):
    csv_files = read_csv(dir_path)
    apt_dict = floow_plan_list(csv_files)
    for apt, floor_plans in apt_dict.items():
        draw_png(csv_files, apt, floor_plans)


if __name__ == "__main__":
    # DIR_PATH = './apt_output/'
    DIR_PATH = './_posts/00CodeNote/project/webscrap_apt/'
    main(DIR_PATH)