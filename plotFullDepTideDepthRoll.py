from workhorseFuncs import *
import warnings
from scipy.io.matlab import _mio5
warnings.filterwarnings("ignore", category=_mio5.MatReadWarning)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

def plot_TideDepthRoll(tides, dat, bdat, tagonI, tagoffI, sloc, ntx=3, l=10, h=5):
    # Set minimalist aesthetic
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['text.color'] = 'black'

    # Slice PRH data
    prhDT = pd.to_datetime(dat['DT'][tagonI:tagoffI])
    prhRoll = dat['roll'][tagonI:tagoffI]
    prhDepth = dat['p'][tagonI:tagoffI]

    # Ensure tides is a DataFrame and its datetime column is datetime
    if not isinstance(tides, pd.DataFrame):
        tides = pd.DataFrame(tides)
    tides['DT'] = pd.to_datetime(tides['DT'])

    # Subset tide data to match PRH time range
    tide_slice = tides[(tides['DT'] >= prhDT[0]) & (tides['DT'] <= prhDT[-1])]
    tideDT = tide_slice['DT']
    prhTHeight = tide_slice['Prediction']

    # Create the figure and subplots
    fig, (ax_tides, ax_roll, ax_depth) = plt.subplots(3, 1, figsize=(l, h), dpi=200, sharex=True,
                                                      constrained_layout=True)

    axes = [(ax_tides, tideDT, prhTHeight, 'Tidal Height'),
            (ax_roll, prhDT, prhRoll, 'Roll (degrees)'),
            (ax_depth, prhDT, prhDepth, 'Depth (m)')]

    for ax, x, y, ylabel in axes:

        if ylabel == 'Tidal Height':
            ax.plot(x, y, color='black', linewidth=0.25, zorder=1)
            ax.fill_between(x, y, 4.5, where=(y > 4.5), color='mediumseagreen', alpha=0.5, zorder=0)
        elif ylabel == 'Roll (degrees)':
            ax.plot(x, y, color='black', linewidth=0.75, zorder=1)
            for start, end in zip(bdat['sTime'], bdat['eTime']):
                ax.axvspan(start, end, color='lightgray', alpha=0.5, zorder=0)

        elif ylabel == 'Depth (m)':
            ax.plot(x, y, color='black', linewidth=0.25, zorder=1)
            ax.fill_between(x, 0, y, color='lightblue', alpha=0.5, zorder=0)
            ax.set_xlim(x[0], x[-1])

        ax.set_ylabel(ylabel)


        # Manually set y-axis ticks
        num_ticks_y = ntx
        # For depth: force ymin = 0
        if ylabel == 'Depth (m)':
            ymin = 0
            ymax = max(y)
        else:
            ymin = min(y)
            ymax = max(y)

        #ymin, ymax = min(y), max(y)
        y_ticks = np.linspace(ymin, ymax, num_ticks_y)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(y_ticks)

        if ylabel == 'Depth (m)':
            ax.invert_yaxis()

        # Format y-axis spine
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))     # X spine down slightly
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

        # Tick styling
        ax.tick_params(axis='both', which='major',
                       labelsize=12, labelcolor='black',
                       width=1.5, direction='out', length=5)

        # Bold tick labels
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # Remove gridlines
        ax.grid(False)

    ### X-axis formatting on bottom plot only ###
    ax_depth.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    x_ticks = np.linspace(prhDT[0].timestamp(), prhDT[-1].timestamp(), ntx)
    ax_depth.set_xticks(pd.to_datetime(x_ticks, unit='s'))

    for label in ax_depth.get_xticklabels():
        label.set_fontweight('bold')

    plt.xlabel('Time')
    plt.tight_layout()
    fig.savefig(sloc)
    plt.show()

wdir = "C:\\_temp workspace\\1ER-LAT-FEED\\data"
os.chdir(wdir)

tidesdir = wdir+'\\tides\\'
feeddir = wdir+'\\feed\\'
behavesdir = wdir + '\\tags\\behaves\\'

# load in prh file
prh, fname, prhdir = load_prh()
depID = fname.split()[0]
print("Loaded prh file for... "+depID)

onemintides = pd.read_csv(tidesdir+"tides\\tides_"+depID+'.csv')
print("Loaded 1m tides file for... "+depID)

bdata = pd.read_excel(behavesdir +"behaves_"+depID+'.xlsx')
bdata = bdata[bdata['State'] == 'feed'].copy()
bdata['sTime'] = pd.to_datetime(bdata['sTime'], format='%d-%b-%Y %H:%M:%S')
bdata['eTime'] = pd.to_datetime(bdata['eTime'], format='%d-%b-%Y %H:%M:%S')

prh['roll'] = np.degrees(prh['roll'])
prh['DN'] = [float(str(dn).replace("'", "")) for dn in prh['DN']]
prh['DT'] = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in make_DT(prh['DN'])]
prh['DS'] = [dt.strftime('%d-%m-%Y %H:%M') for dt in make_DT(prh['DN'])]
prh['vidDT'] = [dt.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(dt) else np.nan for dt in make_DT(prh['vidDN'])]
tonI = np.where(prh['tagon'] == 1)[0][0]
toffI = np.where(prh['tagon'] == 1)[0][-1]
tonDT = pd.to_datetime(prh['DT'][tonI])
toffDT = pd.to_datetime(prh['DT'][toffI])
print(f"Tag put on: {tonDT} at index {tonI}")
print(f"Tag came off: {toffDT} at index {toffI}")
sloc = tidesdir + "prh-tides\\prh-tides_" + depID + ".csv"
saveloc = feeddir+"plots\\tide+depth+roll\\tides+depth+roll_"+depID+".png"

prh = add_tides(prh, onemintides, sloc)
plot_TideDepthRoll(onemintides, prh, bdata, tonI, toffI, saveloc)
