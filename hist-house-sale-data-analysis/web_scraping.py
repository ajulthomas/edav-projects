

import pandas as pd # type: ignore
import numpy as np

import requests
from bs4 import BeautifulSoup as bs


def extract_sales_price(url):
    # get the html page content via http get requests
    page = requests.get(url)

    # create a BeautifulSOup Object from the html content from the http response
    soup = bs(page.content, "html.parser")


    # extract the table header data
    table_col_names = soup.find("thead").find("tr").findAll("th")

    colnames = []

    # loops through each header cell and extracts the text
    for col in table_col_names:
        # converts the text to lower case 
        # trims whitespace at both ends
        # replaces the whitespaces in between with '_' character
        column_name = ((col.text).lower()).strip().replace(' ', '_')
        colnames.append(column_name)


    colnames


    # list to store the house sale data
    data = []

    # extracts the data rows
    table_rows = soup.find("tbody").findAll("tr")

    # parse each row and store it in the data list, creates a 2D-list/table
    for row in table_rows:
        
        row_data = []

        # extracts all table cells in the current rows
        cells = row.findAll('td')

        # loops through each cell in the row and extract the data
        for cell in cells:
            row_data.append((cell.text).strip())

        data.append(row_data)

    # converts the data to a data frame

    df_raw = pd.DataFrame(data, columns= colnames)

    # format price column
    # €971,535.00  -> 971535
    df_raw['price'] = df_raw['price'].str.replace(',','')
    df_raw['price'] = df_raw['price'].str.replace('€','')

    # converting to numeric values
    df_raw['price'] = df_raw['price'].apply(pd.to_numeric, errors="coerce")

    return df_raw['price'].iloc[:50]
