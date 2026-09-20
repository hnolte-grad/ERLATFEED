import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_tufte_boxplots_py(csv_path, vars_to_plot=None, pitch_vars=None):
    # Load the data
    df = pd.read_csv(csv_path)
    df['depID'] = df['depID'].astype(str)

    # Convert pitch to degrees if specified
    if pitch_vars is not None:
        for col in pitch_vars:
            if col in df.columns:
                df[col] = np.degrees(df[col])

    # Determine variables to plot
    if vars_to_plot is None:
        vars_to_plot = df.select_dtypes(include=[np.number]).drop(columns=['depID'], errors='ignore').columns.tolist()

    # Reshape data to long format
    df_long = df[['depID'] + vars_to_plot].melt(id_vars='depID', var_name='Variable', value_name='Value')

    # Add jitter for scatter points
    np.random.seed(42)
    df_long['Jitter'] = np.random.normal(0, 0.1, size=len(df_long))

    # Map depID to numeric positions
    depID_categories = sorted(df_long['depID'].unique())
    depID_to_x = {dep: i for i, dep in enumerate(depID_categories)}
    df_long['xval'] = df_long['depID'].map(depID_to_x) + df_long['Jitter']

    # Faceted plot with increased height and width
    g = sns.FacetGrid(df_long, col="Variable", col_wrap=2, sharey=False, height=6, aspect=2.5)

    def fast_plot(data, **kwargs):
        ax = plt.gca()
        sns.boxplot(x='depID', y='Value', data=data, ax=ax,
                    color='white', fliersize=0, linewidth=1,
                    boxprops=dict(edgecolor='black'))
        ax.scatter(data['xval'], data['Value'],
                   color='black', alpha=0.5, s=12.5, linewidth=0)

        # Reduce y-axis ticks to 4 and increase font size
        ymin, ymax = ax.get_ylim()
        ax.set_yticks(np.linspace(ymin, ymax, 4))
        ax.tick_params(axis='y', labelsize=16)

    g.map_dataframe(fast_plot)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("depID", "")

    # Fix x-axis tick labels for all subplots
    for ax in g.axes.flatten():
        ax.set_xticks(range(len(depID_categories)))
        ax.set_xticklabels(depID_categories, rotation=45, ha='right', fontsize=16)

    # Add extra padding between subplots
    plt.subplots_adjust(wspace=0.35, hspace=0.5)

    # Tight layout with extra space for labels
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


# Example usage
plot_tufte_boxplots_py(
    "C:/_temp workspace/1ER-LAT-FEED/data/feed/analyzed-feedevents/analyzed-feedevents_MASTER.csv",
    vars_to_plot=[
        "durFP_s", "meanDepth_m", "meanRoll_deg",
        "rngRoll_deg", "meanPitchFP", "rangePitchFP"
    ],
    pitch_vars=["meanPitchFP", "rangePitchFP"]  # convert these to degrees
)
