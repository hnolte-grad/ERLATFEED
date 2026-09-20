import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_depID_boxplots(csv_path, vars_to_plot=None, pitch_vars=None, metric="mean"):
    # Load the data
    df = pd.read_csv(csv_path)
    df['depID'] = df['depID'].astype(str)

    # Convert pitch vars to degrees if specified
    if pitch_vars is not None:
        for col in pitch_vars:
            if col in df.columns:
                df[col] = np.degrees(df[col])

    # Default: plot all numeric variables except depID
    if vars_to_plot is None:
        vars_to_plot = df.select_dtypes(include=[np.number]).drop(columns=['depID'], errors='ignore').columns.tolist()

    # Summarize per deployment
    if metric == "mean":
        df_summary = df.groupby("depID")[vars_to_plot].mean().reset_index()
    elif metric == "median":
        df_summary = df.groupby("depID")[vars_to_plot].median().reset_index()
    else:
        raise ValueError("metric must be 'mean' or 'median'")

    # Grid setup
    nvars = len(vars_to_plot)
    ncols = 6
    nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for ax, var in zip(axes, vars_to_plot):
        sns.boxplot(y=df_summary[var], color="white", width=0.3, fliersize=0, linewidth=2, ax=ax)
        sns.stripplot(y=df_summary[var], color="black", size=6, jitter=True, alpha=0.5, ax=ax)

        # Set 3 y-ticks rounded to 1 decimal
        ymin, ymax = ax.get_ylim()
        yticks = np.linspace(ymin, ymax, 3)
        yticks = np.round(yticks, 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=14)

        # Add clean title
        ax.set_title(var.replace("_", " "), fontsize=16, pad=10)

        # Styling
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel(None)
        ax.set_xticks([])

    # Remove unused subplots
    for ax in axes[nvars:]:
        ax.remove()

    plt.tight_layout(pad=2.0)
    plt.show()


# Example usage
plot_depID_boxplots(
    "C:/_temp workspace/1ER-LAT-FEED/data/feed/analyzed-feedevents/analyzed-feedevents_MASTER.csv",
    vars_to_plot=["durFP_s", "meanDepth_m", "meanRoll_deg", "rngRoll_deg", "meanPitchFP", "rangePitchFP"],
    pitch_vars=["meanPitchFP", "rangePitchFP"]
)
