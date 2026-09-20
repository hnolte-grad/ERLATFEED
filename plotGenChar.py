"""
========================================================================================================================
Script Name: plotGenChar.py
Author: Hannah Clayton
Created: 05.08.2025
Description:
    plots for ERLATFEED manuscript showing the variation in kinematics of feeding events
Usage:
Dependencies:
    - Python >= 3.12
Notes:
========================================================================================================================
"""
from workhorseFuncs import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def makeKDE(dat):
    dat = np.array(dat)
    kde = gaussian_kde(dat)
    x = np.linspace(dat.min(), dat.max(), 500)
    y = kde(x)
    return x, y

# SET UP WORKSPACE
ddir = 'C:\\_temp workspace\\1ER-LAT-FEED\\data\\feed\\analyzed-feedevents\\'                                          # data directory
os.chdir(ddir)
fdat = pd.read_csv(ddir+"analyzed-feedevents_MASTER.csv")

# Set up figure
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2)  # 2 rows, 2 columns
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

# Make first subplot with two KDEs, one for entering and one for leaving feed position
d1 = np.array(fdat['durEntFP_s'])
d2 = np.array(fdat['durExFP_s'])
kde1 = gaussian_kde(d1)
kde2 = gaussian_kde(d2)
x1 = np.linspace(min(d1.min(), d2.min()) - 0.5,
                     max(d1.max(), d2.max()) + 0.5, 500)
y1 = kde1(x1)
y2 = kde2(x1)

# make second kde, for next feeding event
cleanLastFE = fdat['lastFE_s'].dropna().values
x2,y3 = makeKDE(cleanLastFE)

# make third kde, for duration in feeding event
x3,y4 = makeKDE(fdat['durFP_s'])


ax1.plot(x1, y1, color='green')
ax1.plot(x1,y2, color='red')
ax1.set_title('Duration (s) Entering +\nLeaving Feeding Position')

ax3.plot(x2, y3)
ax3.set_title('Duration (s) to Next Feed Event')

ax2.plot(x3, y4)
ax2.set_title('Duration (s) in Feed Position')

#
# ax2.plot(x, y2, color='green')
# ax2.set_title('Plot 2: Cosine')
#
# ax3.plot(x, y3, color='red')
# ax3.set_title('Plot 3: Tangent')
#
# # Layout
# plt.tight_layout()
# plt.show()
# fig, (ax_depth, ax_tides, ax_roll) = plt.subplots(3, 1, sharex=True, figsize=(10, 4), constrained_layout=True)
#
#
#
# # Example data
# data1 =
# data2 = np.array(fdat['durExFP_s'])
#
# # KDE estimation
# kde1 = gaussian_kde(data1)
# kde2 = gaussian_kde(data2)
#
# # Create range for plotting

#
# # Evaluate KDEs

#
# # Plot
# plt.plot(x_vals, y1, label='Enter Feed Position', color='green')
# plt.plot(x_vals, y2, label='Exit Feed Position', color='red')
# #plt.fill_between(x_vals, y1, alpha=0.3, color='blue')
# #plt.fill_between(x_vals, y2, alpha=0.3, color='green')
#
# # Customize
# plt.title('Density Plot of Two Datasets')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True)
plt.show()
#
#
# # Dummy data
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
# y3 = np.tan(x)
#
# # Create figure and axes

#
# # First row: 2 plots

