# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file into a pandas dataframe
# df = pd.read_csv('./apt_output/apt_20230325.csv')

# # Plot a line graph of the house prices over time
# plt.plot(df['Floor_plan'], df['Rent'])
# plt.title('Apt Prices')
# plt.xlabel('Floor_plan')
# plt.ylabel('Rent')
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # Read the CSV file
# df = pd.read_csv('./apt_output/apt_20230326.csv')

# # Clean the 'Rent' column by removing non-numeric characters
# df['Rent'] = df['Rent'].str.replace('$','').str.replace(',','').str.extract('(\d+\.?\d*)', expand=False).astype(float)

# # Create a line plot of rent prices over the index
# df.plot(y='Rent', figsize=(10,6))
# plt.title('Rent Price Trend')
# plt.xlabel('Apartment Index')
# plt.ylabel('Rent Price ($)')
# plt.show()


# 3

# import pandas as pd
# import matplotlib.pyplot as plt

# # Read the CSV file
# df = pd.read_csv('./apt_output/apt_20230326.csv')

# # Pivot the data to create a separate column for each floor plan
# pivot_df = df.pivot(index='Date', columns='Floor_plan', values='Rent')

# # Mimic the data for the coming 7 days
# dates = pd.date_range('20230326', periods=8).strftime('%Y%m%d')
# for date in dates[1:]:
#     new_df = pivot_df.loc['20230326'].copy()
#     new_df.name = date
#     pivot_df = pivot_df.append(new_df)

# # Plot the data
# pivot_df.plot(figsize=(10, 6))

# # Add labels and title
# plt.xlabel('Date')
# plt.ylabel('Rent')
# plt.title('Rent Price Trend')

# # Show the plot
# plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt

# # list of CSV files to read
# csv_files = ["./apt_output/apt_20230326.csv", "./apt_output/apt_20230327.csv", "./apt_output/apt_20230328.csv"]

# # list of floor plans to plot
# floor_plans = ["S01", "S01L", "S02", "A03", "A05", "A06", "A07", "A09", "A10", "A12", "A14D", "A16D", "B03"]

# # create empty dataframe to store all data
# all_data = pd.DataFrame()

# # loop through each CSV file and append to all_data
# for file in csv_files:
#     df = pd.read_csv(file, parse_dates=["Date"])
#     all_data = all_data.append(df)

# # filter data by floor plans
# filtered_data = all_data[all_data["Floor_plan"].isin(floor_plans)]

# # loop through each floor plan and plot on the same graph
# for floor_plan in floor_plans:
#     data = filtered_data[filtered_data["Floor_plan"] == floor_plan]

#     # plt.plot(data["Date"], data["Rent"].str.extract(r'\$(.*)/month').astype(float), label=floor_plan)
#     # plt.plot(data["Date"], data["Rent"].str.extract(r'\$(.*)/month').str.replace(',', '').astype(float), label=floor_plan)
#     # plt.plot(data["Date"], data["Rent"].astype(str).str.extract(r'\$(.*)/month').str.replace(',', '').astype(float), label=floor_plan)
#     data["Rent"] = data["Rent"].astype(str).str.replace(',', '').str.extract(r'\$(.*)/month')
#     # print(data["Rent"])

#     # data["Rent"] = data["Rent"].astype(str).str.extract(r'\$(.*)/month').str.replace(',', '')

#     # data["Rent"] = data["Rent"].astype(str).str.extract(r'\$(.*)/month').str.replace(',', '')

#     plt.plot(data["Date"], data["Rent"].astype(float), label=floor_plan)



# # set axis labels and legend
# plt.xlabel("Date")
# plt.ylabel("Rent ($)")
# plt.legend()

# # show the plot
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Define the list of CSV files to read
files = ["./apt_output/apt_20230326.csv", "./apt_output/apt_20230327.csv", "./apt_output/apt_20230328.csv"]

# Define the list of floor plans to plot
floor_plans = ["S01", "S01L", "S02", "A03", "A05", "A06", "A07", "A09", "A10", "A12", "A14D", "A16D", "B03"]

# Create a new figure to plot the data
fig, ax = plt.subplots()

# Loop through each CSV file
for i, file in enumerate(files):
    # Read the data from the CSV file into a Pandas DataFrame
    df = pd.read_csv(file)

    # Loop through each floor plan and plot the data for that floor plan
    for j, floor_plan in enumerate(floor_plans):
        # Subset the DataFrame to only include rows for the current floor plan
        subset = df[df['Floor_plan'] == floor_plan]

        # Plot the data for the current floor plan
        ax.plot(subset['Date'], subset['Rent'], label=floor_plan, linestyle='-', marker='o')

    # Add a legend to the plot for the current CSV file
    ax.legend(title=file, loc='upper left', bbox_to_anchor=(1, 1))

    # Save the plot as a PNG file
    plt.savefig(f'apt_{i}.png', dpi=300)

    # Clear the plot for the next CSV file
    ax.clear()



# import pandas as pd
# import matplotlib.pyplot as plt

# # Define the CSV files to read
# csv_files = ['./apt_output/apt_20230326.csv', './apt_output/apt_20230327.csv', './apt_output/apt_20230328.csv']

# # # Define a list of colors to use for each floor plan
# # colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']

# # Read each CSV file and concatenate them into a single dataframe
# df = pd.concat([pd.read_csv(f) for f in csv_files])

# # Convert the "Date" column to a datetime object
# df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

# # Loop over each floor plan and plot the rent price trend on a line graph
# for i, fp in enumerate(df['Floor_plan'].unique()):
#     fp_df = df[df['Floor_plan'] == fp]

#     fp_df["Rent"] = fp_df["Rent"].astype(str).str.replace(',', '').str.extract(r'\$(.*)/month')
#     plt.plot(fp_df['Date'], fp_df['Rent'], label=fp)

# # Set the title, x-axis label, and y-axis label of the graph
# plt.title('Rent Price Trend')
# plt.xlabel('Date')
# plt.ylabel('Rent Price')

# # Add a legend to the graph
# plt.legend()

# # Show the graph
# plt.show()
