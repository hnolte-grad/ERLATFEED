from os import listdir
from workhorseFuncs import *
import pandas as pd
import glob
from plotly.subplots import make_subplots

def loaddata(fdir, wdir):
    inp = input("Enter [1] for prh-feed file or [2] for analyzed-feed...")
    feprh= "\\prh-feedevents\\prh-feedevents_MASTER.csv"
    afeed = "\\analyzed-feedevents\\analyzed-feeddevents_MASTER.csv"

    fname = feprh if inp == "1" else afeed

    try:
        dat = pd.read_csv(fdir+fname)
        print("Loaded "+fname+"...")
    except FileNotFoundError:
        if inp == "1":
            pth = wdir+'\\feed\\'+fname
            flist = glob.glob(pth + "/*.csv")
            dlist = []
            for file in flist:
                dlist.append(pd.read_csv(file))
            dat = pd.concat(dlist, ignore_index=True)
            dat.to_csv((fdir + 'prh-feedevents\\prh-feedevents_MASTER.csv'), index=False)
            dat['iFeedEvent'] = dat.groupby('feedID').cumcount() + 1
            print("Data combined into master file, indexed, and saved.")

    return dat, inp

def makeprrrpplot(dat):
    # plt.style.use('dark_background')
    # Create subplots with 5 rows, 1 column (no shared x-axis for simplicity)
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    lw = 0.06
    for i, (feedID, group) in enumerate(dat.groupby('feedID')):
        # Reset time_seconds for each feedID to start from 0
        group['time_seconds'] = (group['time_seconds'] - group['time_seconds'].iloc[0])

        # Plot each group on separate subplots
        axs[0].plot(group['time_seconds'], group['depth'], color='black', linewidth=lw)
        axs[1].plot(group['time_seconds'], group['roll'], color='black', linewidth=lw)
        axs[2].plot(group['time_seconds'], group['rollrate'], color='black', linewidth=lw)
        axs[3].plot(group['time_seconds'], group['pitch'], color='black', linewidth=lw)
        # axs[4].plot(group['time_seconds'], group['sjerk'], color='black', linewidth=lw)

    # Set titles, labels, and gridlines for all subplots
    axs[0].set_title('Depth', color='black')
    axs[0].set_ylabel('(m)', color='black')
    # axs[0].set_ylim(0, 10)
    axs[0].invert_yaxis()  # Invert the y-axis for Depth plot
    axs[0].grid(True, color='gray', linestyle='dotted', linewidth=0.5)

    axs[1].set_title('Roll', color='black')
    axs[1].set_ylabel('(deg)', color='black')
    # axs[1].set_ylim(-10, 180)
    axs[1].grid(True, color='gray', linestyle='dotted', linewidth=0.5)

    axs[2].set_title('Rate of Roll', color='black')
    axs[2].set_ylabel('(delta roll/sec-1)', color='black')
    axs[2].set_ylim(-1, 360)
    axs[2].grid(True, color='gray', linestyle='dotted', linewidth=0.5)

    axs[3].set_title('Pitch', color='black')
    axs[3].set_ylabel('(radians)', color='black')
    # axs[3].set_ylim(0.05, 0.06)
    axs[3].grid(True, color='gray', linestyle='dotted', linewidth=0.5)
    axs[3].set_xlabel('Time (seconds)', color='black')

    # axs[4].set_title('Sjerk', color='black')
    # axs[4].set_ylabel('Sjerk', color='black')
    # axs[4].set_ylim(0, 4)
    # axs[4].grid(True, color='gray', linestyle='.', linewidth=0.5)

    max_time = dat['time_seconds'].max()  # Maximum time in seconds
    for ax in axs:
        ax.set_xlim([0, max_time])

    # Set x-axis tick intervals to 15 seconds
    plt.xticks(range(0, int(max_time) + 1, 30))

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()

    # Show the plot
    plt.show()

## -----------------------------------execute code------------------------------------------ ##
wDir = "D:\\_research\\dissertation\\1ER-LAT-FEED\\data"
os.chdir(wDir)

behavesDir = wDir + '\\tags\\behaves\\'
tidesDir = wDir+'\\tides\\'
feedDir = wDir+'\\feed\\'

pfMast, usrInp = loaddata(feedDir, wDir)
pltDat = pfMast['time_seconds', 'depth', 'roll', 'rollrate', 'pitch', 'sjerk', 'feedID']
makeprrrpplot(pltDat) if usrInp == '1' else print('skip')


