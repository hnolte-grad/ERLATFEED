import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set global style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['text.color'] = 'black'

# Load data
file_path = "C:/_temp workspace/1ER-LAT-FEED/data/tides/onemintides/2024-onemintides.csv"
tide_data = pd.read_csv(file_path)

# Time formatting
tide_data['TimeStr'] = tide_data['Time'].astype(str).str[:5]
tide_data['TimeOnly'] = pd.to_datetime(tide_data['TimeStr'], format="%H:%M")
tide_data['DateTime'] = pd.to_datetime(
    tide_data['Date'].astype(str) + ' ' + tide_data['TimeStr'],
    format="%Y/%m/%d %H:%M",
    utc=True
)

# Define season
tide_data['SpringSeason'] = tide_data['Date'].apply(
    lambda x: "Sounder Season (Mar-May)" if "03-01" <= x[5:] <= "05-31" else "Other"
)

# Plot setup
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Plot black lines first (Other)
for date, group in tide_data.groupby('Date'):
    if group['SpringSeason'].iloc[0] == 'Other':
        ax.plot(group['TimeOnly'], group['Pred'], color='black', alpha=0.2, zorder=1)

# Plot blue lines on top (Sounder Season)
for date, group in tide_data.groupby('Date'):
    if group['SpringSeason'].iloc[0] == 'Sounder Season (Mar-May)':
        ax.plot(group['TimeOnly'], group['Pred'], color='darkblue', alpha=0.5, zorder=2)

# Axis limits
xmin = tide_data['TimeOnly'].min()
xmax = tide_data['TimeOnly'].max()
ymin = tide_data['Pred'].min()
ymax = tide_data['Pred'].max()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Format x-axis as HH:MM
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

# Tick configuration
ntx = 3
x_ticks = pd.date_range(start=xmin, end=xmax, periods=ntx)
y_ticks = np.linspace(ymin, ymax, ntx)

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

ax.tick_params(axis='both', which='major', labelsize=12,
               labelcolor='black', width=1.5, direction='out', length=5)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Spine cleanup
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# Optional title
ax.set_title("Tidal Predictions by Day", fontweight='bold')

plt.tight_layout()
plt.show()
