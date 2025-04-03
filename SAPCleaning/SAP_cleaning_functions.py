import os
import re
import pandas as pd

def save_clean_dataframes(folder_name, df_dict):
    """
    Saves DataFrames into a directory, ensuring they contain only the 'Equipment_ID' column.
    
    Parameters:
    - folder_name (str): Name of the directory where the files will be saved.
    - df_dict (dict): Dictionary with names as keys and DataFrames as values.
    
    Returns:
    - None (Saves the files and prints messages to the console).
    """

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Iterate over each DataFrame in the dictionary
    for name, df in df_dict.items():
        # Clean file name to avoid invalid characters
        safe_name = re.sub(r'[\/:*?"<>|]', '_', name)

        # If df is not a DataFrame, convert it to an empty one with 'Equipment_ID'
        if not isinstance(df, pd.DataFrame):
            print(f"⚠️ {name} was not a DataFrame, converting to empty DataFrame with 'Equipment_ID'.")
            df = pd.DataFrame(df, columns=["Equipment_ID"])

        # If the DataFrame has no columns, add 'Equipment_ID'
        if df.shape[1] == 0:
            df["Equipment_ID"] = None  # Add empty column

        # If the DataFrame has more than one column, keep only 'Equipment_ID'
        if "Equipment_ID" in df.columns:
            df = df[["Equipment_ID"]]
        else:
            df.insert(0, "Equipment_ID", None)  # Add the column at the first position

        # Check if the DataFrame is still empty after adjustments
        if df.empty:
            print(f"⚠️ Not saved: {name} (Empty DataFrame)")
        else:
            # Save the DataFrame to CSV
            file_path = os.path.join(folder_name, f"{safe_name}.csv")
            df.to_csv(file_path, index=False)
            print(f"✅ Saved: {file_path}")

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_flagged_equipment(df, hospital_col="Hospital", type_col="Type",
                                 order_col="Last_order_year", purchase_col="Purchase_year",output_name = "flagged_equipment.png"):
    """
    Generates visualizations for flagged equipment with count labels on bars.
    """

    # Create the overall figure
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    # 1. Top 10 Hospitals
    hospital_counts = df[hospital_col].value_counts().head(10)
    bars1 = axs[0, 0].bar(hospital_counts.index, hospital_counts.values, color="steelblue")
    axs[0, 0].set_title("Top 10 Hospitals with Flagged Equipment")
    axs[0, 0].set_xlabel("Hospital")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 0].grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        axs[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), int(bar.get_height()),
                      ha='center', va='bottom')

    # 2. Top 10 Equipment Types
    type_counts = df[type_col].value_counts().head(10)
    bars2 = axs[0, 1].bar(type_counts.index, type_counts.values, color="darkorange")
    axs[0, 1].set_title("Top 10 Equipment Types Flagged")
    axs[0, 1].set_xlabel("Equipment Type")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].tick_params(axis='x', rotation=45)
    axs[0, 1].grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars2:
        axs[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), int(bar.get_height()),
                      ha='center', va='bottom')

    # 3. Last Order Year Distribution
    sns.histplot(df[order_col].dropna(), bins=20, kde=True, ax=axs[1, 0], color="seagreen")
    axs[1, 0].set_title("Last Order Year for Flagged Equipment")
    axs[1, 0].set_xlabel("Year of Last Order")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].set_xlim(2000, 2025)
    axs[1, 0].set_xticks(range(2000, 2026, 2))
    axs[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

    # 4. Purchase Year Distribution
    sns.histplot(df[purchase_col].dropna(), bins=20, kde=True, ax=axs[1, 1], color="purple")
    axs[1, 1].set_title("Purchase Year for Flagged Equipment")
    axs[1, 1].set_xlabel("Year of Purchase")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].set_xlim(2005, 2025)
    axs[1, 1].set_xticks(range(2005, 2026, 2))
    axs[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    output_folder = os.path.join("..", "images")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_name)
    plt.savefig(output_path, dpi=300)
    plt.show()
