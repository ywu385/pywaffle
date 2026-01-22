#%%
from corporateviz import *

#%%
# Initialize once
viz = CorporateViz(
    palette=['#2C3E50', '#E74C3C', '#3498DB', '#F1C40F'], 
    font_family='DejaVu Sans'
)


# Create Data
df = pd.DataFrame({
    'Metric': [f'KPI Category {i}' for i in range(1, 11)],
    'Value': [520, 480, 450, 410, 390, 350, 310, 290, 250, 210]
})

# Define Specific Fonts
my_fonts = {
    'title': {'fontsize': 22},
    'axis_label': {'fontsize': 8, 'color': 'grey'},     # Small categories
    'value_label': {'fontsize': 11, 'fontweight': 'bold', 'color': 'black'} # Pop the numbers
}

fig, ax = viz.plot_barh(
    df=df, x_col='Value', y_col='Metric',
    title="KPI Performance Ranking",
    subtitle="Top 10 metrics by volume",
    font_dict=my_fonts
)
plt.show()


# Create Data
dates = pd.date_range('2024-01-01', periods=8, freq='ME')
df_t = pd.DataFrame({'Date': dates, 'Sales': [100, 110, 105, 130, 150, 160, 155, 170]})

# Create Pointer
notes = [{
    'x': pd.Timestamp('2024-05-31'), 
    'y': 150, 
    'text': "Marketing Push", 
    'offset': (-40, 40) # 40px Left, 40px Up
}]

fig, ax = viz.plot_timeseries(
    df=df_t, date_col='Date', value_col='Sales',
    title="Monthly Sales", 
    annotations=notes
)
plt.show()


viz.plot_sankey(
    labels=["Budget", "R&D", "Marketing", "Salaries"],
    source=[0, 0, 0], # Budget -> R&D, Budget -> Mkt, Budget -> Sal
    target=[1, 2, 3],
    value=[1000, 400, 200, 400],
    title="2024 Budget Flow",
    static_file="budget_flow.png" # Saves directly to file
)


font_settings = {
    # 1. The Main Headline
    'title': {
        'fontsize': 20,
        'fontweight': 'bold',
        'color': '#000000'
    },
    
    # 2. The Context/Description
    'subtitle': {
        'fontsize': 14,
        'style': 'italic',
        'color': '#666666'
    },
    
    # 3. The Y-Axis Categories (e.g., "Department Name")
    # Note: Only supports 'fontsize' and 'color' due to matplotlib limitations
    'axis_label': {
        'fontsize': 9, 
        'color': '#333333'
    },
    
    # 4. The Numbers on the bars (e.g., "500")
    'value_label': {
        'fontsize': 10,
        'fontweight': 'bold',
        'color': '#1B9CFC' # Match brand color?
    }
}

#%%
import pandas as pd
import numpy as np # Make sure you have numpy imported in your notebook
from corporateviz import *
# Setup
viz = CorporateViz(
    palette=['#2C3E50', '#E74C3C', '#3498DB', '#F1C40F', '#2ECC71'], 
    font_family='Arial'
)

# Dummy Data
df_donut = pd.DataFrame({
    'Region': ['North America', 'Europe', 'Asia Pacific', 'LatAm', 'Others'],
    'Revenue': [3500, 2800, 2100, 800, 400]
})

# # Plot
# fig, ax = viz.plot_donut(
#     df=df_donut,
#     cat_col='Region',
#     val_col='Revenue',
#     title="Global Revenue Distribution",
#     subtitle="North America and Europe drive 65% of total volume"
# )

# plt.show()


viz = CorporateViz(
    palette=['#2C3E50', '#E74C3C', '#3498DB', '#9B59B6', '#2ECC71'], 
    font_family='DejaVu Sans'
)

# Custom Font Dict
# We want the labels to be Bold and slightly larger
my_fonts = {
    'title': {'fontsize': 25},
    'value_label': {'fontsize': 18, 'fontweight': 'bold'} # <--- Controls the donut labels
}

# Plot
fig, ax = viz.plot_donut(
    df=df_donut,
    cat_col='Region',
    val_col='Revenue',
    title="Revenue by Region",
    start_angle=45,
    font_dict=my_fonts,
    show_center_label=False
)

plt.show()

#%%

# 1. Create Dummy Data
df_stack = pd.DataFrame({
    'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
    'Hardware': [120, 130, 110, 150],
    'Software': [80, 90, 120, 140],
    'Services': [40, 50, 60, 80]
})

# 2. Setup (if not already done)
viz = CorporateViz(
    palette=['#2C3E50', '#E74C3C', '#3498DB', '#F1C40F'], 
    font_family='DejaVu Sans'
)

# 3. Vertical Stacked Bar
fig, ax = viz.plot_stacked_bar(
    df=df_stack,
    x_col='Quarter',
    stack_cols=['Hardware', 'Software', 'Services'], # <--- The columns to stack
    title="Quarterly Revenue Mix",
    subtitle="Software segment showing strongest growth"
)
plt.show()

# 4. Horizontal Stacked Bar
fig, ax = viz.plot_stacked_barh(
    df=df_stack,
    y_col='Quarter',
    stack_cols=['Hardware', 'Software', 'Services'],
    title="Revenue Composition",
    subtitle="Q4 delivered highest total volume"
)
plt.show()