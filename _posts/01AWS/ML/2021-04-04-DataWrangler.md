---
title: AWS - ML - Data Wrangler
date: 2021-04-04 11:11:11 -0400
categories: [01AWS, ML]
tags: [AWS, ML]
toc: true
image:
---

- [Data Wrangler](#data-wrangler)
  - [basic](#basic)
  - [benefit](#benefit)
  - [Prerequisites](#prerequisites)
  - [Access Data Wrangler](#access-data-wrangler)
  - [Update Data Wrangler](#update-data-wrangler)
- [Demo](#demo)
  - [Demo: Data Wrangler Titanic Dataset Walkthrough](#demo-data-wrangler-titanic-dataset-walkthrough)
    - [Upload Dataset to S3 and Import](#upload-dataset-to-s3-and-import)
    - [Data Flow](#data-flow)
      - [Prepare and Visualize](#prepare-and-visualize)
        - [Data Exploration](#data-exploration)
        - [Drop Unused Columns](#drop-unused-columns)
        - [Clean up Missing Values](#clean-up-missing-values)
        - [Custom Pandas: Encode](#custom-pandas-encode)
      - [Custom SQL: SELECT Columns](#custom-sql-select-columns)
    - [Export](#export)
      - [Export to Data Wrangler Job Notebook](#export-to-data-wrangler-job-notebook)
      - [Training XGBoost Classifier](#training-xgboost-classifier)
      - [Shut down Data Wrangler](#shut-down-data-wrangler)


---

# Data Wrangler

> AWS re:Invent 2020: Accelerate data preparation with Amazon SageMaker Data Wrangler

- ref
  - [AWS re:Invent 2020: Accelerate data preparation with Amazon SageMaker Data Wrangler](https://www.youtube.com/watch?v=_bsat_2N8LI&t=1s&ab_channel=AWSEvents)

---

## basic


![Screen Shot 2021-04-03 at 13.54.02](https://i.imgur.com/FIWDhpz.png)

challengs:
1. data preparation is time consuming and required multiple tools and tasks
2. simple tasks require a lot of code
3. deplotment can require a code rewrite, and productionizing can take months

## benefit


![Screen Shot 2021-04-03 at 13.56.15](https://i.imgur.com/3tep4VB.png)

![Screen Shot 2021-04-03 at 13.57.11](https://i.imgur.com/8RQJjKv.png)


![Screen Shot 2021-04-03 at 13.57.49](https://i.imgur.com/l3DvHKv.png)


![Screen Shot 2021-04-03 at 13.58.39](https://i.imgur.com/CZ8MaOq.png)


![Screen Shot 2021-04-03 at 13.59.28](https://i.imgur.com/zzcZa7L.png)

![Screen Shot 2021-04-03 at 14.00.36](https://i.imgur.com/5aay02p.png)



---

## Prerequisites

To use Data Wrangler, must do the following:

1. need access to a `m5.4xlarge` ec2 instance.
2. Configure the required permissions.
3. To use Data Wrangler, need an active SageMaker Studio instance.
4. When the Studio instance is **Ready**, Access Data Wrangler.

---


## Access Data Wrangler

**To access Data Wrangler in Studio:**

1. to launch Studio, select **Open Studio**.

2. When Studio opens, create a new flow
   - select the **+** sign on the **New data flow** card under **ML tasks and components**.
     - creates a new folder in Studio with a `.flow file` inside, which contains the data flow.
     - The `.flow file` automatically opens in Studio.
   - selecting **File**, then **New**, and choosing **Flow** in the top navigation bar.

3. (Optional) Rename the new folder and the `.flow file`.

4. When create a new `.flow file` in Studio, may see a message at the top of the Data Wrangler interface that says:
   - **Connecting to engine: Establishing connection to engine...**
   - This message persists as long as the **KernelGateway** app on the **User Details** page is **Pending**.
     - To see the status of this app, in the SageMaker console on the **Amazon SageMaker Studio** page, select the name of the user are using to access Studio.
     - On the **User Details** page, see a **KernelGateway** app under **Apps**.
   - Wait until this app status is **Ready** to start using Data Wrangler.
   - This can take around 5 minutes the first time launch Data Wrangler.

5. To get started, choose a data source and use it to import a dataset.
   - When import a dataset, it appears in the data flow.

6. After import a dataset, Data Wrangler automatically infers the type of data in each column.
   - Choose **+** next to the **Data types** step and select **Edit data types**.
   - After add transforms to the **Data types** step, cannot bulk-update column types using **Update types**.

7. Use the data flow to add transforms and analyses.

8. To export a complete data flow, choose **Export** and choose an export option.

9.  Finally, choose the **Components and registries** icon, and select **Data Wrangler** from the dropdown list to see all `.flow file`s you've created.
    - You can use this menu to find and move between data flows.


use Data Wrangler to create an ML data prep flow.


---


## Update Data Wrangler

- It is recommended that periodically update the Data Wrangler Studio app to access the latest features and updates.
- The data wrangler app name starts with **sagemaker-data-wrang**.


---

# Demo

## Demo: Data Wrangler Titanic Dataset Walkthrough

- have already followed the steps in Access Data Wrangler
- have a new data flow file open that intend to use for the demo.
- uses the [Titanic dataset](https://www.openml.org/d/40945). This data set contains the survival status, age, gender, and class (which serves as a proxy for economic status) of passengers aboard the maiden voyage of the RMS Titanic in 1912.

In this tutorial, you:

- Upload the [Titanic dataset](https://www.openml.org/d/40945) to Amazon Simple Storage Service (Amazon S3), and then import this dataset into Data Wrangler.

- Analyze this dataset using Data Wrangler analyses.

- Define a data flow using Data Wrangler data transforms.

- Export the flow to a Jupyter Notebook that can use to create a Data Wrangler job.

- Process the data, and kick off a SageMaker training job to train a XGBoost Binary Classifier.


### Upload Dataset to S3 and Import

- download the [Titanic dataset](https://www.openml.org/d/40945) and upload it to an S3 bucket
- Upload the dataset to an S3 bucket in the same AWS Region want to use to complete this demo.
- When the dataset has been successfully uploaded to Amazon S3, it can import it into Data Wrangler.

**Import the Titanic dataset to Data Wrangler**

1. Select the **Import** tab in the Data Wrangler flow file.

2. Select **Amazon S3** > **Import a dataset from S3**
   - find the bucket to which added the Titanic dataset.
   - Choose the Titanic dataset CSV file to open the **Details** pane.

3. Under **Details**, the **File type** should be CSV.
   - Choose **Add header to table** to specify that the first row of the dataset is a header.

4. Select **Import dataset**.
   - When the dataset is imported into Data Wrangler, it appears in the data flow.
   - view the data flow at any time by selecting the **Prepare** tab.

![import-titanic-dataset](https://i.imgur.com/UPcccTF.png)


### Data Flow

In the data flow section, you’ll notice that the only steps in the data flow are the recently imported dataset and a **Data type** step. After applying transformations, can come back to this tab see what the data flow looks like. Now, add some basic transformations under the **Prepare** and **Analyze** tabs.

#### Prepare and Visualize

- Data Wrangler has built-in <font color=red> transformations and visualizations </font>
  - use to analyze, clean, and transform the data.
- In the **Prepare** tab, all built-in transformations are listed in the right panel, which also contains an area in which can add custom transformations.

to use these transformations.

##### Data Exploration

create a table summary of the data using an analysis:
1. Choose the **+** next to the **Data type** step in the data flow and select **Add analysis**.

2. In the **Analysis** area, select **Table summary** from the dropdown list.

3. Give the table summary a **Name**.

4. Select **Preview** to preview the table that will be created.

5. Choose **Create** to save it to the data flow. It appears under **All Analyses**.

6. Using the statistics, can make observations similar to the following about this dataset:

   - Fare average (mean) is around $33, while the max is over $500. This column likely has outliers.

   - This dataset uses _?_ to indicate missing values. A number of columns have missing values: _cabin_, _embarked_, and _home.dest_

   - The age category is missing over 250 values.


7. Choose **Prepare** to go back to the data flow.
8. Next, clean the data using the insights gained from these stats.



##### Drop Unused Columns

Using the analysis from the previous section, clean up the dataset to prepare it for training. To add a new transform to the data flow, choose **+** next to the **Data type** step in the data flow and choose **Add transform**.

First, drop columns that don't want to use for training.
- use [Pandas](https://pandas.pydata.org/) data analysis library
- or use one of the built-in transforms.

To do this using Pandas:

1. In the **Custom Transform** section, select **Python (Pandas)** from the dropdown list.

2. Enter the following in the code box.

    ```py
    cols = ['name', 'ticket', 'cabin', 'sibsp', 'parch', 'home.dest','boat', 'body']
    df = df.drop(cols, axis=1)
    ```

3. Choose **Preview** to preview the change and then choose **Add** to add the transformation.



To use the built-in transformations:

1. Choose **Manage columns** from the right panel.

2. For **Input column**, choose **cabin**, and choose **Preview**.

3. Verify that the **cabin** column has been dropped, then choose **Add**.

4. Repeat these steps for the following columns: **ticket**, **name**, **sibsp**, **parch**, **home.dest**, **boat**, and **body**.




##### Clean up Missing Values

- do this with the **Handling missing values** transform group.
- A number of columns have missing values. Of the remaining columns, _age_ and _fare_ contain missing values. Inspect this using the **Custom Transform**.

1. Using the **Python (Pandas)** option
   - to quickly review the number of entries in each column: `df.info()`

2. drop rows with missing values in the _age_ category:

   1. Choose **Handling missing values**.

   2. Choose **Drop missing** for the **Transformer**.

   3. Choose **Drop Rows** for the **Dimension**.

   4. Choose _age_ for the **Input column**.

   5. Choose **Preview** to see the new data frame, and then choose **Add** to add the transform to the flow.

   6. Repeat the same process for _fare_.


3. use `df.info()` in the **Custom transform** section to confirm that all rows now have 1,045 values.



##### Custom Pandas: Encode

Try flat encoding using Pandas.
- Encoding categorical data is the process of creating a numerical representation for categories.

- For example, if the categories are Dog and Cat, may encode this information into two vectors: `[1,0]` to represent Dog, and `[0,1]` to represent Cat.


1. In the **Custom Transform** section, choose **Python (Pandas)** from the dropdown list.

2. Enter the following in the code box.

    ```py
    import pandas as pd

    dummies = []
    cols = ['pclass','sex','embarked']
    for col in cols:
        dummies.append(pd.get_dummies(df[col]))

    encoded = pd.concat(dummies, axis=1)

    df = pd.concat((df, encoded),axis=1)
    ```

3. Choose **Preview** to preview the change. The encoded version of each column is added to the dataset.

4. Choose **Add** to add the transformation.




#### Custom SQL: SELECT Columns

Now, select the columns want to keep using SQL. For this demo, select the columns listed in the following `SELECT` statement.
- Because _survived_ is the target column for training, put that column first.


1. In the **Custom Transform** section, select **SQL (PySpark SQL)** from the dropdown list.

2. Enter the following in the code box.

    ```sql
    SELECT survived, age, fare, 1, 2, 3, female, male, C, Q, S
    FROM df;
    ```

3. Choose **Preview** to preview the change.
   - The columns listed in the `SELECT` statement above are the only remaining columns.

4. Choose **Add** to add the transformation.

---

### Export

- a number of export options.
- export to a Data Wrangler job notebook.
  - A Data Wrangler job is used to process the data using the steps defined in the data flow.

![Screen Shot 2021-04-03 at 14.20.15](https://i.imgur.com/chtFLbB.png)


#### Export to Data Wrangler Job Notebook

When export the data flow using a **Data Wrangler job**, a **Jupyter Notebook** is automatically created.
- This notebook automatically opens in the Studio instance
- is configured to run a SageMaker processing job to execute the Data Wrangler data flow (a <font color=blue> Data Wrangler job </font>)

1. Save the data flow. Select **File** and then select **Save Data Wrangler Flow**.

2. Choose the **Export** tab.

3. Select the last step in the data flow: `custom pandas`

4. Choose **Data Wrangler Job**. This opens a Jupyter Notebook.

5. Choose any **Python 3 (Data Science)** kernel for the **Kernel**.

6. When the kernel starts, run the cells in the notebook book until **Kick off SageMaker Training Job (Optional)**.

7. Optionally, can run the cells in **Kick off SageMaker Training Job (Optional)** if want to create a SageMaker training job to train an XGboost classifier.
   - can add the code blocks found in Training XGBoost Classifier to the notebook and run them to use the [XGBoost](https://xgboost.readthedocs.io/en/latest/) open source library to train an XGBoost classifier.

8. Uncomment and run the cell under **Cleanup** and run it to revert the SageMaker Python SDK to its original version.

9. can monitor the `Data Wrangler job status` in the SageMaker console in the **Processing** tab.
    - Additionally, can monitor the Data Wrangler job using Amazon CloudWatch.
    - If kicked off a training job, can monitor its status using the SageMaker console under **Training jobs** in the **Training section**.



#### Training XGBoost Classifier

In the same notebook that kicked off the Data Wrangler job, can pull the data and train a XGBoost Binary Classifier using the prepared data with minimal data preparation.

1. First, upgrade necessary modules using `pip` and remove the `\_SUCCESS` file
   - this last file is problematic when using `awswrangler`

    ```bash
    ! pip install --upgrade awscli awswrangler boto sklearn

    ! aws s3 rm {output_path} \
        --recursive  \
        --exclude "*" \
        --include "*_SUCCESS*"
    ```

2. Read the data from Amazon S3.
   - can use `awswrangler` to recursively read all the CSV files in the S3 prefix.
   - The data is then split into features and labels.
   - The label is the first column of the dataframe.

    ```py
    import awswrangler as wr

    df = wr.s3.read_csv(path=output_path, dataset=True)
    X, y = df.iloc[:,:-1],df.iloc[:,-1]`

    - Finally, create DMatrices (the XGBoost primitive structure for data) and do cross-validation using the XGBoost binary classification.

        `import xgboost as xgb

        dmatrix = xgb.DMatrix(data=X, label=y)

        params = {"objective":"binary:logistic",'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}

        xgb.cv(
            dtrain=dmatrix,
            params=params,
            nfold=3,
            num_boost_round=50,
            early_stopping_rounds=10,
            metrics="rmse",
            as_pandas=True,
            seed=123)
    ```


#### Shut down Data Wrangler

When are finished using Data Wrangler, we recommend shut down the instance it runs on to avoid incurring additional charges.





.
