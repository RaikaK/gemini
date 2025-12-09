import streamlit as st
import pandas as pd
import wandb
import plotly.express as px

# Check if secrets are available, otherwise use empty string (auth might happen via CLI)
# Check if secrets are available, otherwise use empty string (auth might happen via CLI)
# st.set_page_config(layout="wide")

st.title("Penguin Paper Experiment Viewer")

@st.cache_data
def load_experiment_data():
    api = wandb.Api(timeout=60)
    # specified by <entity/project-name>
    runs = api.runs("fkun314/penguin-paper-interviews")

    summary_list, config_list, name_list, tags_list = [], [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        tags_list.append(run.tags if run.tags else [])

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list,
        "tags": tags_list
    })
    
    # Flatten specific columns if needed or just keep them as dicts
    # For easier plotting, we usually want to flatten
    # Let's create a flat dataframe
    
    flat_data = []
    for _, row in runs_df.iterrows():
        item = {"name": row["name"], "tags": row["tags"]}
        item.update(row["config"])
        item.update(row["summary"])
        flat_data.append(item)
        
    return pd.DataFrame(flat_data)

try:
    df = load_experiment_data()
    # Deduplicate columns just in case
    df = df.loc[:, ~df.columns.duplicated()]
    st.write(f"Loaded {len(df)} runs.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar for filtering
st.sidebar.header("Filter Experiments")

# 1. Dynamic Filtering
# Let user select a column to filter by
all_columns = df.columns.tolist()
# Default to likely useful columns
default_filter_col = "tags" if "tags" in all_columns else "name"
if "interviewer_model_name" in all_columns:
    default_filter_col = "interviewer_model_name"

filter_col = st.sidebar.selectbox("Filter Column", all_columns, index=all_columns.index(default_filter_col) if default_filter_col in all_columns else 0)

# Get unique values for the selected column
unique_values = []
if filter_col == "tags":
    # Special handling for list of tags
    # Flatten all lists
    all_tags = []
    for tags in df["tags"]:
        if isinstance(tags, list):
            all_tags.extend(tags)
    unique_values = sorted(list(set(all_tags)))
else:
    unique_values = df[filter_col].astype(str).unique().tolist()
    unique_values.sort()

selected_values = st.sidebar.multiselect(
    f"Select values from {filter_col}",
    unique_values,
    default=unique_values[:min(5, len(unique_values))] # Select first 5 by default
)

if not selected_values:
    st.warning(f"Please select at least one value from {filter_col}.")
    # Show all if nothing selected? Or stop? Let's show all but warn.
    # Actually user usually wants to see something.
    filtered_df = df.copy()
else:
    # Apply Filter
    if filter_col == "tags":
        def tag_filter(row_tags):
            if not isinstance(row_tags, list): return False
            return any(val in row_tags for val in selected_values)
        mask = df['tags'].apply(tag_filter)
        filtered_df = df[mask].copy()
    else:
        # Standard column filter
        filtered_df = df[df[filter_col].astype(str).isin(selected_values)].copy()

st.subheader("Filtered Runs")
st.write(f"Showing {len(filtered_df)} of {len(df)} runs.")
st.dataframe(filtered_df)

# Visualization
st.subheader("Visualization")

# Allow user to select metrics
numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

if numeric_cols:
    # Try to find good defaults
    default_x = 0
    default_y = 0
    
    if "simulation_num" in numeric_cols:
        default_x = numeric_cols.index("simulation_num")
    
    # Heuristic for accuracy
    loss_cols = [c for c in numeric_cols if "eval3_accuracy" in c]
    if loss_cols:
        default_y = numeric_cols.index(loss_cols[0])

    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X Axis", numeric_cols, index=default_x)
    with col2:
        y_axis = st.selectbox("Y Axis", numeric_cols, index=default_y)
    with col3:
        # allow string columns for color
        string_cols = filtered_df.select_dtypes(include=['object', 'string']).columns.tolist()
        color_options = ["name"] + string_cols
        # deduplicate
        possible_colors = sorted(list(set(color_options)))
        default_color = "name"
        if filter_col in possible_colors:
             default_color = filter_col # Color by the filter column by default if possible
        
        color_col = st.selectbox("Color By", possible_colors, index=possible_colors.index(default_color) if default_color in possible_colors else 0)
    
    # Ensure color column exists and fill nans to avoid plotly errors
    if color_col in filtered_df.columns:
         filtered_df[color_col] = filtered_df[color_col].fillna("Unknown")

    fig = px.scatter(
        filtered_df, 
        x=x_axis, 
        y=y_axis, 
        color=color_col, 
        hover_data=["name"] + (["company_name"] if "company_name" in filtered_df.columns else [])
    )
    st.plotly_chart(fig, theme="streamlit") 
else:
    st.info("No numeric columns found for visualization.")

# Grouped Stats
st.subheader("Statistics")
group_col = st.selectbox("Group Statistics By", all_columns, index=all_columns.index(filter_col) if filter_col in all_columns else 0)

if numeric_cols and group_col:
    # Handle list columns (like tags) if selected for grouping?
    # Pandas groupby doesn't like lists.
    # Convert to string just in case
    if filtered_df[group_col].apply(lambda x: isinstance(x, list)).any():
         # Skip or warn? Or convert to str
         filtered_df[f"{group_col}_str"] = filtered_df[group_col].astype(str)
         st.write(filtered_df.groupby(f"{group_col}_str")[numeric_cols].mean())
    else:
         st.write(filtered_df.groupby(group_col)[numeric_cols].mean())
