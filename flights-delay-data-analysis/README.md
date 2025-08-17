# Predicting Airplane Delays

---

## Description

This project aims to predict flight delays due to weather conditions for the busiest airports in the US. The goal is to improve customer experience by providing timely updates on potential delays.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://gitlab.com/data-science-technology-systems/dsts-assignment-2.git

# Navigate to the project directory
cd dsts-assignment-2

```

## Usage

Instructions on how to use the project. Include code examples and screenshots if necessary.

```bash

# Example of how to run the notebook
jupyter-lab onpremise.ipynb

```

## Project Goals

The goals of this notebook are:

1. Process and create a dataset from downloaded ZIP files.
2. Perform exploratory data analysis (EDA).
3. Establish a baseline model and improve it.

## Business Scenario

You work for a travel booking website aiming to improve customer experience by predicting flight delays due to weather. The company wants to inform customers about potential delays when booking flights to or from the busiest US airports.

## Dataset

The dataset contains scheduled and actual departure and arrival times reported by certified US air carriers. It includes data from 2014 to 2018, available in 60 compressed files. Download the data from [this link](https://ucstaff-my.sharepoint.com/:f:/g/personal/ibrahim_radwan_canberra_edu_au/Er0nVreXmihEmtMz5qC5kVIB81-ugSusExPYdcyQTglfLg?e=bNO312).

## Problem Formulation

### Business Problem

Flight delays impact airline operations and customer satisfaction. Our goal is to develop a predictive model to forecast flight delays based on various factors.

### Business Goal

Reduce the rate of flight delays and improve the customer satisfaction and hence the retention rate by at least 20% within the next operational year.

### Machine Learning Problem Statement

Develop a binary classification model to predict flight delays based on historical data and weather factors.

### Success Metrics

- **Accuracy**: Percentage of correct predictions.
- **Precision**: Ratio of true positive predictions to total predicted positives.
- **Recall**: Ratio of true positive predictions to total actual positives.
- **F1 Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Visualize the performance of the classification model.

### Type of ML Problem

This is a supervised binary classification problem.

### Tableau Visualisations

[Tableau Public - Flights Delay Version 2](https://public.tableau.com/views/FlightsDelay_17306247527030/FlightsDelayDashboard?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

[Tableau Public - Flights Delay Version 1](https://public.tableau.com/views/FlightDelays_17303901837140/ExploratoryDataAnalysisofFlightRecords?:language=en-GB&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

### Open a Notebook in Amazon SageMaker JupyterLab

Follow these steps to open a Jupyter Notebook in Amazon SageMaker using the `ml.c5.2xlarge` instance type:

1. **Sign in to the AWS Management Console**:

   - Go to [AWS Management Console](https://aws.amazon.com/console/) and log in with your credentials.

2. **Navigate to SageMaker**:

   - In the search bar at the top, type **SageMaker** and select **Amazon SageMaker** from the list.

3. **Open SageMaker Studio**:

   - If SageMaker Studio is set up, click **Open Studio**.
   - If not, set up SageMaker Studio following [these instructions](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html).

4. **Create or Open a Jupyter Notebook**:

   - In SageMaker Studio, go to **Notebook** > **Create New Notebook**.

5. **Configure the Instance Type**:

   - select **Instance Type** as `ml.c5.2xlarge`.
   - Under **Additional Configurations** increase the used memmory to **25 GB**
   - Provide the git repo url after selecting **Clone a public repository**, under **Git Repositories**.
   - Click on **Create Notebook Instance** to create the notebook.

6. **Start the Notebook**:

   - Click on **Open Jupyter Lab** to open the Lab instance, once it's ready.

7. **Upload CSV Files**:

   - Make sure to upload the required CSV files as specified in the previous sections of this `README.md` file.
   - To upload files:
     - In the file browser of JupyterLab, click the **Upload Files** button (usually represented by an upward arrow icon).
     - Select the CSV files from your local machine and upload them to the appropriate folder in your project directory.

8. **Open the Notebook**:
   - In the JupyterLab file browser, navigate to the folder containing your notebooks and open the desired `.ipynb` file.
