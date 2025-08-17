import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ast
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# function to get only the required columns
def get_required_columns(df_model):
    req_cols = [
        "cost",
        "cost_2",
        "cuisine",
        "votes",
        "type",
        "subzone",
        "rating_text",
    ]
    df_model = df_model[req_cols]
    return df_model


# function to clean the data
def clean_data(df_model):

    # copy the dataframe
    df_model = df_model.copy()

    # Convert string representation of lists to actual lists in the "cuisine" column using .loc[]
    df_model.loc[:, "cuisine"] = df_model["cuisine"].apply(ast.literal_eval)

    # Convert string representation of lists to actual lists in the "type" column, replacing NaN with empty lists, using .loc[]
    df_model.loc[:, "type"] = df_model["type"].fillna("[]").apply(ast.literal_eval)

    # let's clean the subzones column
    # we will extract the last part of the subzone column seperated by commas, if any and remove trailing whitespaces
    df_model.loc[:, "suburb"] = df_model["subzone"].str.split(",").str[-1]

    # remove trailing whitespaces
    df_model.loc[:, "suburb"] = df_model["suburb"].str.strip()

    # drop the subzone column
    df_model.drop("subzone", axis=1, inplace=True)

    # remove columns where rating text is not available
    df_model = df_model.dropna(subset=["rating_text"])
    df_model = df_model.dropna(subset=["type"])

    # let's simplify the rating_text column into binary classes
    # we will encode 'Good', 'Very Good', 'Excellent' as 1 and 'Average', 'Poor' as 0
    df_model["rating_text"] = df_model["rating_text"].apply(
        lambda x: 1 if x in ["Good", "Very Good", "Excellent"] else 0
    )

    return df_model


# function to add new column for each cuisine
# check each row and if the cuisine is presents in the cuisine list
# then set the value to 1 else 0
def add_cuisine_columns(df, top_cuisines):
    for cuisine in top_cuisines:
        key = cuisine.lower().replace(" ", "_")
        df[key] = df["cuisine"].apply(lambda x: 1 if cuisine in x else 0)
    return df


# function to add new column for each type
# check each row and if the type is presents in the type list
# then set the value to 1 else 0
def add_type_columns(df, top_types):
    for type in top_types:
        key = type.lower().replace(" ", "_")
        df[key] = df["type"].apply(lambda x: 1 if type in x else 0)
    return df


# function to feature engineer the data
def feature_engineer_data(df_model):
    # x and y split
    X = df_model.drop("rating_text", axis=1)
    y = df_model["rating_text"]

    # test train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # handling missing values in cost and cost_2
    # fill missing values with the median of the column
    cost_median = X_train["cost"].median()
    cost_2_median = X_train["cost_2"].median()

    X_train["cost"] = X_train["cost"].fillna(cost_median)
    X_train["cost_2"] = X_train["cost_2"].fillna(cost_2_median)

    X_test["cost"] = X_test["cost"].fillna(cost_median)
    X_test["cost_2"] = X_test["cost_2"].fillna(cost_2_median)

    # list to store all the cusines to obtain unique cuisines
    cuisine_list = []
    for cuisine in X_train["cuisine"]:
        cuisine_list.extend(cuisine)

    # convert the list to a pandas series
    cuisine_series = pd.Series(cuisine_list)

    # get unique cuisines
    cuisine_list = cuisine_series.unique()

    # cuisines that are served in less than 100 restaurants
    # will be considered as other
    cuisine_counts = cuisine_series.value_counts()

    # get the cuisines that are served in more than 100 restaurants
    top_cuisines = cuisine_counts[cuisine_counts >= 100].index

    # get the cuisines that are served in less than 100 restaurants
    other_cuisines = cuisine_counts[cuisine_counts < 100].index

    # create new column called other_cuisines,
    # set the value of the column to 1 if any of the cuisines in the other_cuisines list is present
    # in the list in the cuisine column, else set the value to 0
    X_train["other_cuisines"] = X_train["cuisine"].apply(
        lambda x: 1 if any(cuisine in other_cuisines for cuisine in x) else 0
    )
    X_test["other_cuisines"] = X_test["cuisine"].apply(
        lambda x: 1 if any(cuisine in other_cuisines for cuisine in x) else 0
    )

    # add the cuisine columns
    add_cuisine_columns(X_train, top_cuisines)
    add_cuisine_columns(X_test, top_cuisines)

    # drop the cuisine column
    X_train = X_train.drop("cuisine", axis=1)
    X_test = X_test.drop("cuisine", axis=1)

    # let's aggregate the type column, i.e merge all lists in type column into one list
    type_list = []
    for type in X_train["type"]:
        type_list.extend(type)

    # convert the list to a pandas series
    type_series = pd.Series(type_list)

    # get unique types
    type_list = type_series.unique()

    # type counts
    type_counts = type_series.value_counts()

    # get the types that are highest in number
    top_types = type_counts[type_counts >= 100].index

    # get the types that are lowest in number
    other_types = type_counts[type_counts < 100].index

    # create new column called other_types,
    # set the value of the column to 1 if any of the types in the other_types list is present
    # in the list in the type column, else set the value to 0
    X_train["other_types"] = X_train["type"].apply(
        lambda x: 1 if any(type in other_types for type in x) else 0
    )
    X_test["other_types"] = X_test["type"].apply(
        lambda x: 1 if any(type in other_types for type in x) else 0
    )

    # add the type columns
    add_type_columns(X_train, top_types)
    add_type_columns(X_test, top_types)

    # drop the type column
    X_train = X_train.drop("type", axis=1)
    X_test = X_test.drop("type", axis=1)

    # check the number of suburbs that appear less than 2 times
    # these suburbs will be considered as other suburbs
    suburb_counts = X_train["suburb"].value_counts()
    other_suburbs = suburb_counts[suburb_counts < 2].index

    # check the number of suburbs that appear more than 2 times
    top_suburbs = suburb_counts[suburb_counts >= 2].index

    # create new column called other_suburbs
    # set the value of the column to 1 if the suburb is in the other_suburbs list, else set the value to 0
    X_train["other_suburbs"] = X_train["suburb"].apply(
        lambda x: 1 if x in other_suburbs else 0
    )
    X_test["other_suburbs"] = X_test["suburb"].apply(
        lambda x: 1 if x in other_suburbs else 0
    )

    # create new column for top suburbs and set the value to 1 if the suburb is in the top suburbs list
    # else set the value to 0
    for suburb in top_suburbs:
        key = suburb.lower().replace(" ", "_")
        X_train[key] = X_train["suburb"].apply(lambda x: 1 if x == suburb else 0)
        X_test[key] = X_test["suburb"].apply(lambda x: 1 if x == suburb else 0)

    # drop the suburb column
    X_train = X_train.drop("suburb", axis=1)
    X_test = X_test.drop("suburb", axis=1)

    # standardize the data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # use pca to reduce the number of features
    pca = PCA(n_components=0.98)
    pca.fit(X_train_scaled)

    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    X_train_pca = pd.DataFrame(X_train_pca)
    X_test_pca = pd.DataFrame(X_test_pca)

    return X_train_pca, X_test_pca, y_train, y_test


# main function to perform the data cleaning and preprocessing
def classifier_model():

    # load the data
    df = load_data("./data/zomato_df_final_data.csv")

    # get the required columns
    df_model = get_required_columns(df)

    # clean the data
    df_model = clean_data(df_model)

    # feature engineer the data
    X_train_pca, X_test_pca, y_train, y_test = feature_engineer_data(df_model)

    # modelling
    # classifier model
    model_classification_3 = LogisticRegression(max_iter=1000).fit(X_train_pca, y_train)

    # predict the test data
    y_test_pred = model_classification_3.predict(X_test_pca)

    # calculate the accuracy of the model
    train_accuracy = model_classification_3.score(X_train_pca, y_train)
    test_accuracy = model_classification_3.score(X_test_pca, y_test)

    print("Test accuracy model_classification_3 : ", test_accuracy)

    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # plot the confusion matrix with value 0 corresponds to "class 1" and 1 corresponds to "class 2"
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Confusion Matrix for Logistic Regression")
    # plt.show()

    # print the confusion matrix
    print("\n")
    print("Confusion Matrix for Logistic Regression")
    print("\n")
    print(conf_matrix)
