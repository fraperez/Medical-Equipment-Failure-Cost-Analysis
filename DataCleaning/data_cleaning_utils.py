

def basic_data_checks(df, remove_duplicates=True):
    """
    Perform basic data quality checks on the DataFrame:
    - Checks for missing values and prints details only if any exist.
    - Checks for duplicate rows and optionally removes them.
    - Identifies columns with constant values (single unique value).
    - Prints the DataFrame shape and data types.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be checked.
    remove_duplicates : bool, optional (default=True)
        If True, duplicate rows will be dropped from the DataFrame.
        
    Returns:
    --------
    pd.DataFrame
        The DataFrame after performing the checks and cleaning (if remove_duplicates is True).
    """
    print("Initial DataFrame shape:", df.shape)
    
    # Check for missing values
    total_missing = df.isna().sum().sum()
    if total_missing == 0:
        print("No missing values found in the DataFrame.")
    else:
        print(f"Total missing values in the DataFrame: {total_missing}")
        missing_per_column = df.isna().sum()
        missing_per_column = missing_per_column[missing_per_column > 0]
        print("Missing values per column:")
        print(missing_per_column)
        
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count == 0:
        print("No duplicate rows found in the DataFrame.")
    else:
        print(f"Found {duplicate_count} duplicate rows in the DataFrame.")
        if remove_duplicates:
            df.drop_duplicates(inplace=True)
            print("Duplicates removed.")
            print("New DataFrame shape:", df.shape)
    
    # Check for constant columns (columns with a single unique value)
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print("\nColumns with constant values (these may not be informative):")
        print(constant_cols)
    
    # Print data types for each column
    print("\nData types of columns:")
    print(df.dtypes)
    
    return df
## Column Cleaning Script: 'Fabricante' and 'Modelo'
# This script is designed to be used separately, taking a DataFrame as an argument.

import pandas as pd
import re

import pandas as pd


import pandas as pd

def add_dollar_day_price(main_df, 
                         dolar_df, 
                         date_col_main="Fecha", 
                         date_col_dolar="Fecha", 
                         price_col_dolar="Precio", 
                         new_col="Precio dolar dia"):
    """
    Adds a column to main_df with the daily dollar price.
    
    For each row in main_df, the function:
      - Converts the date columns to datetime.
      - Checks if the date is less than the minimum date in dolar_df.
          If so, assigns a price of 3.
      - Otherwise, finds the matching date (or the nearest date)
        in dolar_df and takes the corresponding price.
    
    Parameters:
    - main_df: The primary DataFrame containing the day values.
    - dolar_df: The DataFrame that has the date and the corresponding price.
    - date_col_main: Name of the date column in main_df.
    - date_col_dolar: Name of the date column in dolar_df.
    - price_col_dolar: Name of the column in dolar_df that contains the price.
    - new_col: Name of the new column to add to main_df.
    
    Returns:
    - A new DataFrame with the added column "Precio dolar dia".
    """
    
    # Ensure the date columns are in datetime format
    main_df[date_col_main] = pd.to_datetime(main_df[date_col_main])
    dolar_df[date_col_dolar] = pd.to_datetime(dolar_df[date_col_dolar])
    
    # Sort both DataFrames by their date columns (required for merge_asof)
    main_df = main_df.sort_values(date_col_main)
    dolar_df = dolar_df.sort_values(date_col_dolar)
    
    # Get the minimum date in dolar_df
    min_date = dolar_df[date_col_dolar].min()
    
    # Merge the main_df with dolar_df using merge_asof to get the nearest date match
    merged = pd.merge_asof(main_df, 
                           dolar_df[[date_col_dolar, price_col_dolar]], 
                           left_on=date_col_main, 
                           right_on=date_col_dolar, 
                           direction="nearest")
    
    # Create the new column with the price from dolar_df
    merged[new_col] = merged[price_col_dolar]
    
    # For rows where the main date is earlier than the minimum date in dolar_df, assign 3
    merged.loc[merged[date_col_main] < min_date, new_col] = 3
    
    return merged

