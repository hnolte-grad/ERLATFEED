# Calculate feeding rates relative to # of tidal hours above the initiate feeding threshold.
# Mean tidal height @ initiation of feeding = 7.7 ft
# Psuedo-code
# read in 1m prh tides, and feeding event files
# calculate the duration of time between that mean tidal height (count the number of rows, read in the fs, calculate
# num of seconds relative to the fs. return the date string that that tidal height occurred at, count the number of
# feeding events occurring between these

import os
import pandas as pd

def getTidalFeedRate(tdat, thrsh, prhfs):

    # get the total number of hours of the deployment that occur over the tidal threshold
    indsAboveThrsh= tdat[tdat['Prediction'] >= thrsh].shape[0]
    hrsThresh = indsAboveThrsh/((prhfs*60)*60)

    aboveThreshInd = []
    belowThreshInd = []
    flag1 = True
    flag2 = False

    # for i in range(1, len(tdat)-1):
    #         if tdat.loc[i,'Prediction'] >= thrsh and tdat.loc[i-1,'Prediction'] < tdat.loc[i,'Prediction']:
    #             if not flag1:
    #                 aboveThreshInd.append(tdat.loc[i, 'DT'])
    #             flag1 = True
    #             flag2 = False
    #         elif thrsh <= tdat.loc[i, 'Prediction'] < tdat.loc[i - 1, 'Prediction']:
    #             if not flag2:
    #                 belowThreshInd.append(tdat.loc[i, 'DT'])
    #             flag1 = False
    #             flag2 = True
    for i in range(0, len(tdat)):
        if thrsh <= tdat['Prediction'][i]:
            if flag1:
                aboveThreshInd.append(tdat.loc[i, 'DT'])
                flag1 = False
                flag2 = True
        elif thrsh >= tdat['Prediction'][i]:
            if flag2:
                belowThreshInd.append(tdat.loc[i, 'DT'])
                flag2 = False
                flag1 = True




    # Output the results
    print("To add to rate document:\n")
    print(f"Hours above feeding threshold: {hrsThresh}")

    print("Passes threshold at: ")
    print(aboveThreshInd)

    print("\nBelow thresh at: ")
    print(belowThreshInd)

    print(f"Tag on: {tdat['DT'].iloc[0]}")  # First DT in DataFrame
    print(f"Tag off: {tdat['DT'].iloc[-1]}")  # Last DT in DataFrame
    return aboveThreshInd, belowThreshInd
def countFeeds(row):
    return feeds.loc[(feeds['startDT'] >= row['above']) & (feeds['startDT'] <= row['below']), 'pitcount'].sum()

depID = 'er240407-71'
wdir = 'C:\\_temp workspace\\1ER-LAT-FEED\\data\\'
os.chdir(wdir)

prhtides = pd.read_csv(f"{wdir}tides\\prh-tides\\prh-tides_{depID}.csv")
feeds = pd.read_csv(f"{wdir}feed\\analyzed-feedevents\\analyzed-feedevents_{depID}.csv")
thresh = 4.5 # threshold for "feeding tide"
fs = 10 # sampling frequency
feeds['startDT'] = pd.to_datetime(feeds['startDT'])

#reachthrsh, fallthrsh = getTidalFeedRate(prhtides, thresh, fs)

ttimes = pd.read_csv(f"{wdir}tides\\tiderates\\tiderates_{depID}.csv")
ttimes['above'] = pd.to_datetime(ttimes['above'])
ttimes['below'] = pd.to_datetime(ttimes['below'])

ttimes['pitcount']= ttimes.apply(countFeeds, axis=1)
print(ttimes)