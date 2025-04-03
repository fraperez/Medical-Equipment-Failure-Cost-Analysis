import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def combined_analysis_plot(
    df_equipment, 
    df_orders, 
    order_class=None,
    weighted=False,
    metric='count',
    stacked_by_class=False, 
    repair_option="no",     
    title="Analysis Chart", 
    xlabel="Equipment Age (years)", 
    ylabel=None,
    as_percentage=False
):
    """
    Combined analysis and visualization function for maintenance order data.

    Parameters:
    - df_equipment: DataFrame containing equipment info (must include 'Equipment_ID' and 'Purchase_year').
    - df_orders: DataFrame containing service orders.
    - order_class: Optional. If provided, filters orders by this class.
    - weighted: If True, calculates exposure and rate.
    - metric: 'count' for number of orders, 'cost' for sum of costs.
    - stacked_by_class: If True and order_class is None, stack bars by order class.
    - repair_option: 'repair', 'no repair', or 'no'. Filters orders based on OrderWithRepair.
    - title, xlabel, ylabel: Chart customization.
    - as_percentage: Normalize stacked bars to show proportions instead of raw values.
    """

    if order_class is not None:
        df_filtered = df_orders[df_orders['Class'] == order_class].copy()
    else:
        df_filtered = df_orders.copy()

    if repair_option.lower() in ["repair", "no repair"]:
        if "OrderWithRepair" not in df_filtered.columns:
            raise ValueError("Missing 'OrderWithRepair' column in df_orders.")
        df_filtered = df_filtered[df_filtered["OrderWithRepair"] == (repair_option.lower() == "repair")]

    if metric not in ['count', 'cost']:
        raise ValueError("'metric' must be 'count' or 'cost'.")

    if weighted:
        max_year = df_filtered['Year'].max()
        if pd.isnull(max_year):
            raise ValueError("No valid data in 'Year' column after filtering.")

        rows = []
        for _, eq in df_equipment.iterrows():
            purchase_year = eq['Purchase_year']
            max_age = max_year - purchase_year
            if max_age < 0:
                continue
            for age in range(max_age + 1):
                rows.append((eq['Equipment_ID'], age))

        df_exposure = pd.DataFrame(rows, columns=['Equipment_ID', 'Age'])

        exposed = df_exposure.groupby('Age').size().reset_index(name='Exposed_equipment')

        # Aggregation and merging
        if repair_option.lower() == "repair":
            group_cols = ['Equipment_age', 'OrderWithRepair']
            col_to_use = 'Cost' if metric == 'cost' else None
        elif stacked_by_class and order_class is None:
            group_cols = ['Equipment_age', 'Class']
            col_to_use = 'Cost' if metric == 'cost' else None
        else:
            group_cols = ['Equipment_age']
            col_to_use = 'Cost' if metric == 'cost' else None

        if col_to_use:
            agg = df_filtered.groupby(group_cols)[col_to_use].sum().reset_index(name='Value')
        else:
            agg = df_filtered.groupby(group_cols).size().reset_index(name='Value')

        agg.rename(columns={'Equipment_age': 'Age'}, inplace=True)
        df_merged = pd.merge(exposed, agg, on='Age', how='left').fillna(0)

        if len(group_cols) > 1:
            df_merged['Rate'] = df_merged['Value'] / df_merged['Exposed_equipment']
            pivot = df_merged.pivot(index='Age', columns=group_cols[1], values='Rate').fillna(0)
            if 'OrderWithRepair' in pivot.columns:
                pivot = pivot.reindex(columns=[True, False])
        else:
            df_merged['Rate'] = df_merged['Value'] / df_merged['Exposed_equipment']
            pivot = df_merged.set_index('Age')[['Rate']]

    else:
        # Unweighted version
        if repair_option.lower() == "repair":
            group_cols = ['Equipment_age', 'OrderWithRepair']
        elif stacked_by_class and order_class is None:
            group_cols = ['Equipment_age', 'Class']
        else:
            group_cols = ['Equipment_age']

        col_to_use = 'Cost' if metric == 'cost' else None

        if col_to_use:
            agg = df_filtered.groupby(group_cols)[col_to_use].sum().reset_index(name='Value')
        else:
            agg = df_filtered.groupby(group_cols).size().reset_index(name='Value')

        agg.rename(columns={'Equipment_age': 'Age'}, inplace=True)

        if len(group_cols) > 1:
            pivot = agg.pivot(index='Age', columns=group_cols[1], values='Value').fillna(0)
            if 'OrderWithRepair' in pivot.columns:
                pivot = pivot.reindex(columns=[True, False])
        else:
            pivot = agg.set_index('Age')[['Value']]

    if as_percentage:
        pivot = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
        if ylabel is None:
            ylabel = "Proportion of Orders"
    else:
        if ylabel is None:
            if weighted:
                ylabel = "Failure Rate" if metric == 'count' else "Cost Rate"
            else:
                ylabel = "Number of Orders" if metric == 'count' else "Total Cost"

    fig, ax = plt.subplots(figsize=(12, 4))
    if len(pivot.columns) > 1:
        pivot.plot(kind='bar', stacked=True, ax=ax, alpha=0.8, edgecolor='black')
    else:
        col_name = pivot.columns[0]
        ax.bar(pivot.index, pivot[col_name], color='steelblue', alpha=0.7, edgecolor='black')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 20)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_yticklabels([])
    plt.tight_layout()
    output_folder = os.path.join("..", "images")
    os.makedirs(output_folder, exist_ok=True)
    plot_filename = "CostRate_by_age.png" 
    plt.savefig(os.path.join(output_folder, plot_filename), dpi=300)
    plt.show()

    return fig, ax
    

def analyze_orders_by_model(
    orders_df, 
    equipment_df,
    model_col='Model', 
    Equipment_ID='Equipment_ID',
    orderwithrepair_col='OrderWithRepair', 
    year_col='Year', 
    year_start=2010, 
    year_end=2024, 
    min_equip_count=2,
    title=None,
    n_top=10,
    weighted=False):
    """
    Analiza y grafica la Count de órdenes asociadas a diferentes modelos en un período determinado,
    distinguiendo las órdenes con 'OrderWithRepair' True y False en un gráfico apilado.
    
    Además, imprime:
      - El total de órdenes (o la métrica ponderada) en el período.
      - Un ranking de modelos con detalles (Type, Subtype, Manufacturer, Model y Count).
      
    Parameters:
      - orders_df: DataFrame con información de órdenes.
      - equipment_df: DataFrame con información de equipamiento, que debe incluir 'Equipment_ID' y,
                      preferiblemente, 'Type', 'Subtype', 'Manufacturer' y la columna especificada en model_col.
      - model_col: Nombre de la columna en equipment_df que indica el Model (default 'Model').
      - Equipment_ID: Nombre de la columna que conecta ambos DataFrames (default 'Equipment_ID').
      - orderwithrepair_col: Nombre de la columna en orders_df que indica si la orden incluyó reparación (True/False).
      - year_col: Nombre de la columna que indica el Year de la orden (default 'Year').
      - year_start: Year inicial para filtrar órdenes (default 2010).
      - year_end: Year final para filtrar órdenes (default 2024).
      - title: Título opcional para el gráfico.
      - n_top: Limita el número de modelos principales a mostrar (default 10).
      - weighted: Si True, normaliza la métrica de Orders per la Count de equipos para ese Model.
      
    Returns:
      - pivot_df: DataFrame pivotado con el Number of Orders por Model, diferenciado por 'OrderWithRepair'.
    """
   # 1. Filtrar Orders per el período especificado.
    orders_df = orders_df[orders_df["Class"] == "ICOR"]
    orders_period = orders_df[(orders_df[year_col] >= year_start) & (orders_df[year_col] <= year_end)]

    # 2. Verificar que la columna de enlace exista en ambos DataFrames.
    if Equipment_ID not in orders_period.columns:
        print(f"La columna '{Equipment_ID}' no está en orders_df.")
        return
    if Equipment_ID not in equipment_df.columns:
        print(f"La columna '{Equipment_ID}' no está en equipment_df.")
        return

    # 3. Definir columnas a incluir del equipment_df (debe incluir el Model).
    merge_cols = [Equipment_ID, model_col]
    # Si están disponibles, agregar columnas adicionales para más detalles.
    for col in ["Type", "Subtype", "Manufacturer"]:
        if col in equipment_df.columns:
            merge_cols.append(col)
        elif col.capitalize() in equipment_df.columns:
            merge_cols.append(col.capitalize())

    # 4. Realizar merge para incorporar el Model y detalles adicionales.
    merged_df = orders_period.merge(equipment_df[merge_cols], on=Equipment_ID, how='left')
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # 5. Agrupar por Model y el indicador de reparación, contando el Number of Orders.
    group_df = merged_df.groupby([model_col, orderwithrepair_col]).size().reset_index(name='orders')

    # 6. Pivotear para tener los modelos como índice y columnas según el valor de OrderWithRepair.
    pivot_df = group_df.pivot(index=model_col, columns=orderwithrepair_col, values='orders').fillna(0)

    # 7. Reordenar columnas si hay exactamente dos (True y False).
    if pivot_df.shape[1] == 2 and set(pivot_df.columns) == {True, False}:
        pivot_df = pivot_df.reindex(columns=[True, False])

    # 8. Calcular el total de Orders per Model y ordenar de forma descendente.
    pivot_df['total'] = pivot_df.sum(axis=1)
    total_by_model = pivot_df['total']
    pivot_df = pivot_df.drop(columns='total')

    # 9. Filtrar por Count mínima de equipos antes de ordenar los top n.
    equip_counts = equipment_df[model_col].value_counts()
    valid_models = equip_counts[equip_counts >= min_equip_count].index

    # Filtrar solo modelos que existen en total_by_model
    valid_models = valid_models.intersection(total_by_model.index)

    # Aplicar filtro
    total_by_model = total_by_model.loc[valid_models]
    pivot_df = pivot_df.loc[valid_models]


    # 10. Si weighted es True, normalizar por el conteo de equipos para cada Model.
    if weighted:
        weighted_total = total_by_model.copy()
        for model in weighted_total.index:
            if model in equip_counts and equip_counts[model] != 0:
                weighted_total[model] = weighted_total[model] / equip_counts[model]
                pivot_df.loc[model] = pivot_df.loc[model] / equip_counts[model]
        total_by_model = weighted_total

    # 11. Ordenar nuevamente y limitar a los top n modelos.
    total_by_model = total_by_model.sort_values(ascending=False).head(n_top)
    pivot_df = pivot_df.loc[total_by_model.index]

    

    
    # 11. Imprimir ranking por Model con detalles.
    table_data = []
    for i, (model, total) in enumerate(total_by_model.items(), start=1):
        equipment_details = equipment_df[equipment_df[model_col] == model]
        if not equipment_details.empty and all(col in equipment_details.columns for col in ["Type", "Subtype", "Manufacturer"]):
            # Extraer detalles de la primera ocurrencia.
            type_val = equipment_details.iloc[0]["Type"]
            subtype_val = equipment_details.iloc[0]["Subtype"]
            manufacturer_val = equipment_details.iloc[0]["Manufacturer"]
        else:
            type_val, subtype_val, manufacturer_val = ("", "", "")
        table_data.append({
            "Type": type_val,
            "Subtype": subtype_val,
            "Manufacturer": manufacturer_val,
            "Model": model,
            "Count": round(total, 1)
        })
    print()  # Salto de línea adicional.
    
    # 12. Crear el gráfico de barras apiladas (se mantiene como estaba).
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot_df.plot(
        kind='bar', 
        stacked=True, 
        color=['#1b9e77', '#d95f02'], 
        alpha=0.9, 
        ax=ax
    )
    
    # Agregar etiquetas en las barras.
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=2)
    
    ax.set_xlabel(model_col, fontsize=12)
    ax.set_ylabel("Number of Orders" + (" (Weighted)" if weighted else ""), fontsize=12)
    ax.set_title(title if title else f"Orders per {model_col} ({year_start}-{year_end}) - {orderwithrepair_col} (Min. {min_equip_count} units)", fontsize=14, fontweight='bold')
    ax.legend(title=orderwithrepair_col)
    ax.set_ylim(0, pivot_df.values.max() + pivot_df.values.max()*0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    output_folder = os.path.join("..", "images")
    os.makedirs(output_folder, exist_ok=True)
    plot_filename = f"orders_by_{model_col.lower()}.png"
    plt.savefig(os.path.join(output_folder, plot_filename), dpi=300)
    plt.show()
    
    # 13. Imprimir una tabla formateada con Type, Subtype, Manufacturer, Model y Count.
    table_df = pd.DataFrame(table_data)
    # Utilizamos el método to_string para mostrarla de forma tabular.
    print("Details by Model Table:")
    print(tabulate(table_df, headers="keys", tablefmt="pretty", showindex=False))
    
    return pivot_df



def analyze_data(
    df,
    year_column,
    metric='count',
    split_by_class=False,
    classes=None,
    sum_column="Cost",
    title=None,
    equipment_df=None,
    equipment_year_column="Purchase_year"
):
    """
    Function to analyze equipment or order data over time and generate visualizations.

    Parameters:
    - df: DataFrame containing order data.
    - year_column: Name of the column that contains the year (purchase or order).
    - metric: 'count' to count records, or 'sum' to sum a column.
    - split_by_class: If True, separate data by 'Class' (requires 'Class' column).
    - classes: List of specific classes to include (optional).
    - sum_column: Column to sum if metric is 'sum'. Default is "Cost".
    - title: Title for the plots.
    - equipment_df: DataFrame with equipment data.
    - equipment_year_column: Name of the column in equipment_df with the purchase year.
    """

    # Aggregate by year
    if metric == 'count':
        data_by_year = df.groupby(year_column).size()
    elif metric == 'sum':
        if sum_column in df.columns:
            data_by_year = df.groupby(year_column)[sum_column].sum()
        else:
            raise ValueError(f"Column '{sum_column}' not found in DataFrame.")
    else:
        raise ValueError("Invalid metric. Use 'count' or 'sum'.")

    data_by_year = data_by_year.sort_index()
    cumulative = data_by_year.cumsum()

    # Weighting by equipment count
    if equipment_df is not None and equipment_year_column in equipment_df.columns:
        equipment_counts = {
            year: equipment_df[equipment_df[equipment_year_column] <= year].shape[0]
            for year in data_by_year.index
        }
        equipment_counts_series = pd.Series(equipment_counts)
        data_by_year = data_by_year / equipment_counts_series
        cumulative = data_by_year.cumsum()

    # 1. Cumulative line plot
    plt.figure(figsize=(12, 3.5))
    plt.plot(cumulative.index, cumulative.values, marker="o", linestyle="-", color="black", linewidth=2, label="Cumulative (weighted)")
    plt.xlabel("Year")
    ylabel_text = "Cumulative Count (weighted)" if metric == 'count' else f"Cumulative {sum_column} (weighted)"
    plt.ylabel(ylabel_text)
    plt.title(title)
    plt.gca().set_yticklabels([])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_folder = os.path.join("..", "images")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, "cumulative_plot.png"), dpi=300)
    plt.show()

    # 2. Yearly bar plot
    plt.figure(figsize=(12, 3.5))
    plt.bar(data_by_year.index.astype(str), data_by_year.values, color='dodgerblue', alpha=0.75)
    plt.xlabel("Year", fontsize=12)
    ylabel_text = "Count (weighted)" if metric == 'count' else f"Total {sum_column} (weighted)"
    plt.ylabel(ylabel_text, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.gca().set_yticklabels([])
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "yearly_difference_plot.png"), dpi=300)
    plt.show()

    # 3. Split by OrderWithRepair for ICOR
    if classes == ["ICOR"] and 'OrderWithRepair' in df.columns:
        df_icor = df[df['Class'] == "ICOR"]
        data_by_year_repair = df_icor.groupby([year_column, 'OrderWithRepair']).size().unstack().fillna(0)

        fig, ax = plt.subplots(figsize=(12, 3.5))
        data_by_year_repair.plot(kind='bar', stacked=True, alpha=0.85, ax=ax)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(title='OrderWithRepair', labels=['False', 'True'])
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_yticklabels([])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "repair_order_distribution.png"), dpi=300)
        plt.show()

    # 4. Split by Class
    if split_by_class and 'Class' in df.columns:
        if not classes:
            classes = df['Class'].unique()
        if metric == 'count':
            data_by_year_class = df.groupby([year_column, 'Class']).size().unstack().fillna(0)
        elif metric == 'sum':
            data_by_year_class = df.groupby([year_column, 'Class'])[sum_column].sum().unstack().fillna(0)

        if equipment_df is not None and equipment_year_column in equipment_df.columns:
            print("Weighting by number of active devices...")
            equipment_counts = {
                year: equipment_df[equipment_df[equipment_year_column] <= year].shape[0]
                for year in data_by_year_class.index
            }
            equipment_counts_series = pd.Series(equipment_counts, index=data_by_year_class.index)
            data_by_year_class = data_by_year_class.div(equipment_counts_series, axis=0)

        data_by_year_class = data_by_year_class[[cls for cls in classes if cls in data_by_year_class.columns]]

        # Stacked bar plot by Class
        fig, ax = plt.subplots(figsize=(12, 3.5))
        data_by_year_class.plot(kind="bar", stacked=True, alpha=0.85, ax=ax)
        ax.set_xlabel("Year", fontsize=8)
        ylabel_text = "Count" if metric == 'count' else f"Total {sum_column}"
        ax.set_ylabel(ylabel_text, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_yticklabels([])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "stacked_by_class.png"), dpi=300)
        plt.show()

        # Cumulative line plot by Class
        plt.figure(figsize=(12, 3.5))
        for cls in classes:
            if cls in df['Class'].unique():
                if metric == 'count':
                    class_data = df[df['Class'] == cls].groupby(year_column).size().sort_index()
                elif metric == 'sum':
                    class_data = df[df['Class'] == cls].groupby(year_column)[sum_column].sum().sort_index()

                if equipment_df is not None and equipment_year_column in equipment_df.columns:
                    print("Weighting by number of active devices...")
                    equipment_counts = {
                        year: equipment_df[equipment_df[equipment_year_column] <= year].shape[0]
                        for year in class_data.index
                    }
                    equipment_counts_series = pd.Series(equipment_counts, index=class_data.index)
                    class_data = class_data / equipment_counts_series

                cumulative_class = class_data.cumsum()
                plt.plot(cumulative_class.index, cumulative_class.values, marker="o", linestyle="-", linewidth=1.5, label=f"{cls} Accumulated")

        plt.xlabel("Year")
        ylabel_text = "Accumulated Count" if metric == 'count' else f"Accumulated {sum_column}"
        plt.ylabel(ylabel_text)
        plt.title(title)
        plt.gca().set_yticklabels([])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "cumulative_by_class.png"), dpi=300)
        plt.show()



def plot_dual_axis_yearly_data(df, year_column, cost_column="Cost"):
    """
    Creates a bar chart with two Y-axes to display the number of devices and investment per year,
    with improved colors and contrast.
    
    Parameters:
    - df: DataFrame containing the data.
    - year_column: Name of the column that contains the year.
    - cost_column: Name of the column that contains the investment value (default "Cost").
    """
    # Group data by year: count of devices and total investment
    count_series = df.groupby(year_column).size().sort_index()
    cost_series = df.groupby(year_column)[cost_column].sum().sort_index()
    
    # Get the list of years (ensuring both datasets have the same reference)
    years = sorted(set(count_series.index) | set(cost_series.index))
    
    # Extract values, filling with 0 if a year is missing in one of the groups
    count_values = [count_series.get(year, 0) for year in years]
    cost_values = [cost_series.get(year, 0) for year in years]
    
    # Positions for the bars
    x = np.arange(len(years))
    bar_width = 0.4
    
    # Create figure and dual axes
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    
    color_devices = "#377eb8"
    color_investment = "#e41a1c"
    
    # Plot device count
    bars1 = ax1.bar(x - bar_width/2, count_values, 
                    width=bar_width, 
                    color=color_devices, 
                    alpha=0.9, 
                    label="Number of devices")

    # Plot investment
    bars2 = ax2.bar(x + bar_width/2, cost_values, 
                    width=bar_width, 
                    color=color_investment, 
                    alpha=0.9, 
                    label="Investment")
    
    # Labels and title
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Number of devices", color=color_devices, fontsize=12)
    ax2.set_ylabel("Investment", color=color_investment, fontsize=12)
    ax1.set_yticklabels([])
    ax2.set_yticklabels([])

    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45, ha='right')
    plt.title("Number of Devices and Investment per Year", fontsize=14, fontweight='bold')
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
    
    plt.tight_layout()

    # Save plot
    output_folder = os.path.join("..", "images")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, "dual_axis_yearly_plot.png"), dpi=300)

    plt.show()
def compare_costs_vs_purchase_value_filtered(
    df_equipment,
    df_orders,
    category_filter=None,
    Class=None,
    plot=True
):
    """
    Compare the total cost spent on orders for each equipment against its initial purchase value.
    Returns a DataFrame with equipment for which the total order cost exceeds the purchase value.
    """
    if category_filter is not None:
        for col, value in category_filter.items():
            if col in df_equipment.columns:
                df_equipment = df_equipment[df_equipment[col] == value]
            else:
                print(f"Column '{col}' not found in df_equipment.")

    if Class is not None:
        df_orders = df_orders[df_orders["Class"] == Class]

    df_cost = df_orders.groupby("Equipment_ID")["Cost"].sum().reset_index(name="Total_Order_Cost")
    df_comparison = df_equipment[['Equipment_ID', 'Purchase_value']].merge(df_cost, on="Equipment_ID", how="left")
    df_comparison["Total_Order_Cost"] = df_comparison["Total_Order_Cost"].fillna(0)

    df_comparison["Ratio"] = df_comparison.apply(
        lambda row: row["Total_Order_Cost"] / row["Purchase_value"] if row["Purchase_value"] != 0 else np.nan,
        axis=1
    )

    df_above_line = df_comparison[df_comparison["Ratio"] > 1].copy()

    print(f"Total number of equipment above the line: {len(df_above_line)}")

    if plot:
        plt.figure(figsize=(11, 3))
        df_below = df_comparison[df_comparison["Total_Order_Cost"] <= df_comparison["Purchase_value"]]

        plt.scatter(df_below["Purchase_value"], df_below["Total_Order_Cost"], color='blue', label="≤ Purchase Value")
        plt.scatter(df_above_line["Purchase_value"], df_above_line["Total_Order_Cost"], color='red', label="> Purchase Value")

        plt.xlabel("Initial Purchase Value")
        plt.ylabel("Total Cost Spent on Orders")
        plt.title("Total Order Cost vs. Purchase Value")

        max_val = max(df_comparison["Purchase_value"].max(), df_comparison["Total_Order_Cost"].max())
        plt.plot([0, max_val], [0, max_val], linestyle="--", color="gray", label="Reference Line (y = x)")
        plt.xlim(0, max_val * 0.025)
        plt.ylim(0, max_val * 0.025)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.legend()
        plt.grid(True)

        output_folder = os.path.join("..", "images")
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder, "cost_vs_purchase_value.png"), dpi=300)
        plt.show()

    return df_above_line
def plot_top_expense_percentage_vs_purchase_cost(
    df_equipment,
    df_orders,
    category=None,
    top_n=15,
    min_equipment=1,
    order_type="both",
    min_purchase_value=0
):
    """
    Analyze and visualize the percentage of expenses (ICOR and IPRV orders) relative to the purchase cost,
    displaying the top groups and a summary table with:
      - Type, Subtype, Manufacturer, Model
      - Total spent, number of repairs, number of devices
      - Average purchase cost
      - Percentage = Total spent / (number of devices * avg purchase cost) * 100
    """
    df_equipment = df_equipment[df_equipment['Purchase_value'] >= min_purchase_value]

    if category is None:
        df_equipment_group = df_equipment[['Equipment_ID', 'Purchase_value']].copy()
        df_equipment_group.rename(columns={'Purchase_value': 'total_purchase'}, inplace=True)
        df_equipment_group['equipment'] = 1
    else:
        df_equipment_group = (
            df_equipment
            .groupby(category, as_index=False)
            .agg(
                equipment=('Equipment_ID', 'count'),
                total_purchase=('Purchase_value', 'sum')
            )
        )

    if order_type in ["ICOR", "IPRV"]:
        df_orders = df_orders[df_orders['Class'] == order_type]

    if category:
        merge_cols = ['Equipment_ID'] + category if isinstance(category, list) else ['Equipment_ID', category]
    else:
        merge_cols = ['Equipment_ID']

    df_merged = pd.merge(df_orders, df_equipment[merge_cols], on="Equipment_ID", how="left")

    if category:
        group_cols = category + ['Class'] if isinstance(category, list) else [category, 'Class']
        group_base = category if isinstance(category, list) else [category]
    else:
        group_cols = ['Equipment_ID', 'Class']
        group_base = ['Equipment_ID']

    df_spent_group = (
        df_merged
        .groupby(group_cols)['Cost']
        .sum()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={'ICOR': 'total_ICOR', 'IPRV': 'total_IPRV'})
    )

    df_repairs_count = (
        df_merged
        .groupby(group_base)
        .size()
        .reset_index(name="num_repairs")
    )

    df_spent_group = pd.merge(df_spent_group, df_repairs_count, on=group_base, how="left").fillna(0)

    if order_type == "ICOR":
        df_spent_group['total_spent'] = df_spent_group['total_ICOR']
        df_spent_group['total_IPRV'] = 0
    elif order_type == "IPRV":
        df_spent_group['total_spent'] = df_spent_group['total_IPRV']
        df_spent_group['total_ICOR'] = 0
    else:
        df_spent_group['total_spent'] = df_spent_group['total_ICOR'] + df_spent_group['total_IPRV']

    df_group = pd.merge(df_equipment_group, df_spent_group, on=group_base, how="left").fillna(0)

    df_group['avg_purchase_cost'] = np.where(
        df_group['equipment'] != 0,
        df_group['total_purchase'] / df_group['equipment'],
        0
    )

    df_group['percentage'] = np.where(
        (df_group['equipment'] != 0) & (df_group['avg_purchase_cost'] != 0),
        (df_group['total_spent'] / (df_group['equipment'] * df_group['avg_purchase_cost'])) * 100,
        np.nan
    ).round(1)

    if category:
        df_group = df_group[df_group['equipment'] >= min_equipment]

    df_top = df_group.nlargest(top_n, 'percentage').sort_values('percentage', ascending=False)

    plt.figure(figsize=(12, 4.5))
    bar_width = 0.7
    indices = np.arange(len(df_top))

    pct_ICOR = np.where(
        df_top['avg_purchase_cost'] != 0,
        (df_top["total_ICOR"] / (df_top['equipment'] * df_top['avg_purchase_cost'])) * 100,
        0
    )
    pct_IPRV = np.where(
        df_top['avg_purchase_cost'] != 0,
        (df_top["total_IPRV"] / (df_top['equipment'] * df_top['avg_purchase_cost'])) * 100,
        0
    )

    if order_type == "ICOR":
        bar = plt.bar(indices, pct_ICOR, bar_width, label="ICOR", color="skyblue")
        for i, rect in enumerate(bar):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2, height + 0.5,
                     f"{height:.1f}%", ha="center", va="bottom", fontsize=9)
    elif order_type == "IPRV":
        bar = plt.bar(indices, pct_IPRV, bar_width, label="IPRV", color="salmon")
        for i, rect in enumerate(bar):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2, height + 0.5,
                     f"{height:.1f}%", ha="center", va="bottom", fontsize=9)
    else:
        bar1 = plt.bar(indices, pct_ICOR, bar_width, label="ICOR", color="skyblue")
        bar2 = plt.bar(indices, pct_IPRV, bar_width, bottom=pct_ICOR, label="IPRV", color="salmon")
        total_pct = pct_ICOR + pct_IPRV
        for i, val in enumerate(total_pct):
            plt.text(i, val + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    x_labels = [" - ".join(str(row[col]) for col in category) if isinstance(category, list) else str(row[category])
                 for _, row in df_top.iterrows()] if category else df_top['Equipment_ID'].astype(str)

    plt.xticks(indices, x_labels, rotation=45, ha="right")
    plt.title(f"Top {top_n} {category or 'Equipment_ID'} with Highest Expense % vs. Purchase Cost")
    plt.xlabel(category or 'Equipment_ID')
    plt.ylabel("Expense Percentage (%)")
    plt.legend()
    plt.tight_layout()

    output_folder = os.path.join("..", "images")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, "top_expense_percentage.png"), dpi=300)
    plt.show()

    model_key = 'Model' if 'Model' in df_equipment.columns else 'Model'
    merge_key = category if category else "Equipment_ID"
    extra_cols = [col for col in ["Type", "Subtype", "Manufacturer", model_key, "Purchase_value", "Equipment_ID"] if col in df_equipment.columns]
    df_extra = df_equipment[extra_cols].drop_duplicates()

    df_final = pd.merge(df_top, df_extra, on=merge_key, how="left", suffixes=("", "_dup"))

    df_final['Purchase_value'] = df_final['avg_purchase_cost'].round(1)
    df_final['Total_spent'] = df_final['total_spent'].round(1)
    df_final['Ratio'] = df_final['percentage'].round(1)
    df_final['Repairs'] = df_final['num_repairs'].round(1)
    df_final['N_equipment'] = df_final['equipment'].round(1)

    if category is None:
        display_cols = [col for col in ["Type", "Manufacturer", "Model", "Equipment_ID", "Repairs", "Ratio"] if col in df_final.columns]
    else:
        display_cols = [col for col in ["Type", "Manufacturer", "Model", "N_equipment", "Repairs", "Ratio"] if col in df_final.columns]

    df_table = df_final[display_cols].copy()
    df_table = df_table.drop_duplicates().head(top_n)

    print("\nFinal Table:")
    col_widths = {col: max(df_table[col].astype(str).map(len).max(), len(col)) for col in df_table.columns}
    df_centered = df_table.copy()
    for col in df_table.columns:
        df_centered[col] = df_table[col].astype(str).apply(lambda x: x.center(col_widths[col]))

    print(tabulate(df_centered, headers="keys", tablefmt="psql", showindex=False))

    return df_top

import seaborn as sns
from tabulate import tabulate



def plot_icor_repair_percentage(df_orders, plot_type="both", show_labels=True, show_trendline=False):
    """
    Generates charts to analyze the percentage of 'ICOR' orders that required repair.

    Parameters:
    -----------
    df_orders : DataFrame
        Must contain:
         - 'Equipment_age' (equipment age in years),
         - 'Year' (order year),
         - 'Class' (order type),
         - 'OrderWithRepair' (boolean: 1 if repaired, 0 if not).
    plot_type : str
        'year', 'antiquity', or 'both'.
    show_labels : bool
        Whether to show % labels.
    show_trendline : bool
        Whether to show trendlines.
    """
    df_icor = df_orders[df_orders['Class'] == 'ICOR']

    if plot_type in ["both", "year"]:
        repairs_by_year = df_icor.groupby('Year')['OrderWithRepair'].mean() * 100
        repairs_by_year.index = pd.to_numeric(repairs_by_year.index, errors='coerce')
        repairs_by_year = repairs_by_year.dropna()
        repairs_by_year = repairs_by_year[(repairs_by_year.index >= 2009) & (repairs_by_year.index <= 2024)]
        no_repairs_by_year = 100 - repairs_by_year

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(repairs_by_year.index, repairs_by_year.values, marker='o', label='With Repair')
        ax.plot(no_repairs_by_year.index, no_repairs_by_year.values, marker='o', label='Without Repair')

        if show_labels:
            for i, (x_val, y_val) in enumerate(zip(repairs_by_year.index, repairs_by_year.values)):
                if i % 2 == 0:
                    ax.text(x_val, y_val, f"{y_val:.1f}%")
            for i, (x_val, y_val) in enumerate(zip(no_repairs_by_year.index, no_repairs_by_year.values)):
                if i % 2 == 0:
                    ax.text(x_val, y_val, f"{y_val:.1f}%")

        if show_trendline:
            for series in [(repairs_by_year, 'Repair'), (no_repairs_by_year, 'No Repair')]:
                x, y = series[0].index.values, series[0].values
                x, y = x.astype(float), y.astype(float)
                mask = ~np.isnan(x) & ~np.isnan(y)
                if mask.sum() > 1:
                    z = np.polyfit(x[mask], y[mask], 1)
                    p = np.poly1d(z)
                    ax.plot(x[mask], p(x[mask]), linestyle='dashed', label=f'{series[1]} Trendline')

        ax.set_title('Percentage of ICOR Orders with and without Repair (by Year)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage of Orders (%)')
        ax.set_ylim(0, 100)
        ax.set_xticks(range(2009, 2025))
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("..", "images", "repair_percentage_by_year.png"), dpi=300)
        plt.show()

    if plot_type in ["both", "antiquity"]:
        repairs_by_age = df_icor.groupby('Equipment_age')['OrderWithRepair'].mean() * 100
        repairs_by_age.index = pd.to_numeric(repairs_by_age.index, errors='coerce')
        repairs_by_age = repairs_by_age.dropna()
        repairs_by_age = repairs_by_age[(repairs_by_age.index >= 0) & (repairs_by_age.index <= 19)]
        no_repairs_by_age = 100 - repairs_by_age

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(repairs_by_age.index, repairs_by_age.values, marker='o', label='With Repair')
        ax.plot(no_repairs_by_age.index, no_repairs_by_age.values, marker='o', label='Without Repair')

        if show_labels:
            for i, (x_val, y_val) in enumerate(zip(repairs_by_age.index, repairs_by_age.values)):
                if i % 2 == 0:
                    ax.text(x_val, y_val, f"{y_val:.1f}%")
            for i, (x_val, y_val) in enumerate(zip(no_repairs_by_age.index, no_repairs_by_age.values)):
                if i % 2 == 0:
                    ax.text(x_val, y_val, f"{y_val:.1f}%")

        if show_trendline:
            for series in [(repairs_by_age, 'Repair'), (no_repairs_by_age, 'No Repair')]:
                x, y = series[0].index.values, series[0].values
                x, y = x.astype(float), y.astype(float)
                mask = ~np.isnan(x) & ~np.isnan(y)
                if mask.sum() > 1:
                    z = np.polyfit(x[mask], y[mask], 1)
                    p = np.poly1d(z)
                    ax.plot(x[mask], p(x[mask]), linestyle='dashed', label=f'{series[1]} Trendline')

        ax.set_title('Percentage of ICOR Orders with and without Repair (by Equipment Age)')
        ax.set_xlabel('Equipment Age (years)')
        ax.set_ylabel('Percentage of Orders (%)')
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 20))
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("..", "images", "repair_percentage_by_age.png"), dpi=300)
        plt.show()

def analyze_equipment_data(
    df_equipment,
    df_orders,
    manufacturer_col="Manufacturer",
    cost_col="Cost",
    Add=None,
    top_n=10,
    min_equipment=15,
    min_year=2020,
    max_year=2024,
    year_reference=2025,
    include_high_failure=False,
    high_failure_threshold=95,
    corr_offset=0.5,
    min_lum=0.8
):
    """
    Analyzes failure rate, repair costs, and equipment age per manufacturer or model.

    Returns a dictionary with:
      - Top failure rate
      - Top average cost
      - Age analysis
      - Summary table
      - High failure rate (optional)
    """
    # Filter only ICOR orders with repair in specified year range
    df_orders_filtered = df_orders[
        (df_orders["OrderWithRepair"] == 1) &
        (df_orders["Class"] == "ICOR") &
        (df_orders["Year"] >= min_year) &
        (df_orders["Year"] <= max_year)
    ]

    # Merge orders with equipment to get manufacturer
    df_merged = df_orders_filtered.merge(
        df_equipment[[manufacturer_col, "Equipment_ID"]], on="Equipment_ID", how="left"
    )

    # Failure count = unique equipment with at least one repair
    df_failures = df_merged.groupby(manufacturer_col)["Equipment_ID"].nunique().reset_index(name="Failure_Count")

    # Total number of devices per manufacturer
    df_total = df_equipment.groupby(manufacturer_col)["Equipment_ID"].nunique().reset_index(name="Total_Equipment")

    # Total repair cost
    df_costs = df_merged.groupby(manufacturer_col)[cost_col].sum().reset_index(name="Total_Repair_Cost")

    # Total number of repairs (not just unique equipment)
    df_repairs = df_merged.groupby(manufacturer_col).size().reset_index(name="Total_Repairs")

    # Merge all
    df_final = df_failures.merge(df_total, on=manufacturer_col)\
                          .merge(df_costs, on=manufacturer_col)\
                          .merge(df_repairs, on=manufacturer_col)

    # Filter out low-frequency manufacturers
    df_final = df_final[df_final["Total_Equipment"] >= min_equipment]

    # Metrics
    df_final["Failure_Rate"] = (df_final["Failure_Count"] / df_final["Total_Equipment"]) * 100
    df_final["Avg_Repair_Cost_Per_Device"] = df_final["Total_Repair_Cost"] / df_final["Total_Equipment"]

    # Optional: include models from 'Add'
    text_add = ""
    if Add:
        text_add = " & Added Models"
        df_extra = df_final[df_final["Model"].isin(Add)]
        missing = set(Add) - set(df_extra["Model"])
        if missing:
            df_missing = pd.DataFrame({"Model": list(missing)})
            for col in df_final.columns:
                if col not in df_missing.columns:
                    df_missing[col] = np.nan
            df_extra = pd.concat([df_extra, df_missing], ignore_index=True)
        df_final = pd.concat([df_final, df_extra]).drop_duplicates(subset=["Model"])

    # Top by failure or cost
    df_high_fail = df_final[df_final["Failure_Rate"] > high_failure_threshold]
    if df_high_fail.shape[0] > top_n:
        df_top_failure = df_high_fail.sort_values("Avg_Repair_Cost_Per_Device", ascending=False).head(top_n)
        df_top_cost = df_top_failure.copy()
    else:
        df_top_failure = df_final.sort_values("Failure_Rate", ascending=False).head(top_n)
        df_top_cost = df_final.sort_values("Avg_Repair_Cost_Per_Device", ascending=False).head(top_n)

    # Age analysis
    df_equipment_top = df_equipment[df_equipment[manufacturer_col].isin(df_top_failure[manufacturer_col])]
    df_equipment_top["Age"] = year_reference - df_equipment_top["Purchase_year"]
    df_age = df_equipment_top.groupby(manufacturer_col)["Age"].agg(["mean", "std"]).reset_index()
    df_age.columns = [manufacturer_col, "AvgAge", "StdAge"]

    # Extra details
    agg_info = {
        "Type": "first",
        "Subtype": "first",
        "Manufacturer": "first",
        "Model": "first"
    }
    agg_info = {k: v for k, v in agg_info.items() if k != manufacturer_col}
    df_info = df_equipment.groupby(manufacturer_col, as_index=False).agg(agg_info)

    df_summary = df_top_failure.merge(df_age, on=manufacturer_col, how="left")\
                               .merge(df_info, on=manufacturer_col, how="left")

    df_summary["Failure_Rate"] = df_summary["Failure_Rate"].round(1)
    df_summary["Avg_Repair"] = df_summary["Avg_Repair_Cost_Per_Device"].round(1)
    df_summary["AvgAge"] = df_summary["AvgAge"].round(1)
    df_summary["StdAge"] = df_summary["StdAge"].round(1)

    # Select relevant columns
    if manufacturer_col == "Model":
        df_summary = df_summary[["Type", "Manufacturer", "Model", "Total_Equipment", "Failure_Rate", "Total_Repairs", "Avg_Repair", "AvgAge"]]
    else:
        df_summary = df_summary[["Type", "Total_Equipment", "Failure_Rate", "Total_Repairs", "Avg_Repair", "AvgAge"]]

    df_summary = df_summary.sort_values("Avg_Repair", ascending=False)

    # === Plots ===
    # 1. Failure Rate
    plt.figure(figsize=(12, 4))
    ax1 = sns.barplot(data=df_top_failure, x="Failure_Rate", y=manufacturer_col, palette="Reds_r")
    for patch, color in zip(ax1.patches, sns.color_palette("Reds_r", len(df_top_failure))):
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = 'white' if lum < 0.5 else 'black'
        ax1.annotate(f"{patch.get_width():.1f}", (patch.get_width() - 2, patch.get_y() + patch.get_height() / 2),
                     ha='right', va='center', fontsize=10, color=text_color, fontweight='bold')
    ax1.set_title(f"Top {top_n} {manufacturer_col} by Failure Rate ({min_year}-{max_year}){text_add}")
    ax1.set_xlabel("Failure Rate (%)")
    ax1.set_xlim(0, 100)
    ax1.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join("..", "images", "top_failure_rate.png"), dpi=300)
    plt.show()

    # 2. Avg Repair Cost
    plt.figure(figsize=(12, 4))
    ax2 = sns.barplot(data=df_top_cost, x="Avg_Repair_Cost_Per_Device", y=manufacturer_col, palette="Blues_r")
    for patch, color in zip(ax2.patches, sns.color_palette("Blues_r", len(df_top_cost))):
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = 'white' if lum < 0.6 else 'black'
        offset = corr_offset * patch.get_width() if lum > min_lum else 0
        ##ax2.annotate(f"${patch.get_width():,.1f}",
        ##             (patch.get_width() - (patch.get_width() * 0.02) + offset, patch.get_y() + patch.get_height() / 2),
        ##             ha='right', va='center', fontsize=10, color=text_color, fontweight='bold')
    ax2.set_title(f"Top {top_n} {manufacturer_col} by Avg Repair Cost")
    ax2.set_xlabel("Avg Repair Cost ($)")
    ax2.grid(axis="x", linestyle="--", alpha=0.7)
    ax2.set_xticks([])
    ax2.set_xticklabels([])

    plt.tight_layout()
    plt.savefig(os.path.join("..", "images", "top_avg_repair_cost.png"), dpi=300)
    plt.show()

    # 3. Age plot
    plt.figure(figsize=(12, 4))
    ax3 = plt.gca()
    x_pos = range(len(df_age))
    bars = ax3.bar(x_pos, df_age["AvgAge"], yerr=df_age["StdAge"], capsize=5, alpha=0.7, edgecolor="black")
    for i, (avg, std) in enumerate(zip(df_age["AvgAge"], df_age["StdAge"])):
        ax3.text(i, avg + std + 0.2, f"{avg:.1f} ± {std:.1f}", ha="center", va="bottom", fontsize=9)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(df_age[manufacturer_col], rotation=45, ha="right")
    ax3.set_title("Average Age of Equipment (Top Failure Rate)")
    ax3.set_ylabel("Average Age (years)")
    ax3.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join("..", "images", "top_avg_age.png"), dpi=300)
    plt.show()

    # Print summary table
    print("Failure, Cost and Age Summary Table:")
    df_summary.drop(columns=["Avg_Repair"], inplace=True)
    print(tabulate(df_summary, headers="keys", tablefmt="pretty", showindex=False))

    # Return results
    results = {
        "top_failure_rate": df_top_failure,
        "top_avg_cost": df_top_cost,
        "age_analysis": df_age,
        "summary_table": df_summary
    }
    if include_high_failure:
        results["high_failure_rate"] = df_high_fail

    return df_merged
