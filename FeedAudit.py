"""
Script Name: main_ErFeedAnalysis.py
Description:

Author: Hannah Clayton
Date Created: 2024-09-27
Last Modified: YYYY-MM-DD
Version: 1.0

Usage:

Dependencies:
    - pandas
    - matplotlib
    - numpy
    - os
    - scipy
    - datetime

Notes:
    - Make sure to install the required dependencies before running the script.
    - Adjust the file paths in the script according to your local setup.

Known Issues:

1) Sumtides files when created store the durations as whole integers when they should be floats representing
    decimal days. To resolve open the sumtides file and change to float, save, and rerun script.
TypeError: unsupported operand type(s) for -: 'NoneType' and int
"""



################################## import functions ###################################

from workhorseFuncs import *
import warnings
from scipy.io.matlab import _mio5
warnings.filterwarnings("ignore", category=_mio5.MatReadWarning)

##################################### MAIN  ########################################
#----------------------LOAD IN FILES, SET THE ENVIRONMENT -------------------------#
wdir = "D:\\_research\\dissertation\\1ER-LAT-FEED\\data"
os.chdir(wdir)

behavesdir = wdir + '\\tags\\behaves\\'
tidesdir = wdir+'\\tides\\'
feeddir = wdir+'\\feed\\'

# load in prh file
prh, fname, prhdir = load_prh()
depID = fname.split()[0]
print("Loaded prh file for... "+depID)

# load in behavioral state files
behaves = pd.read_excel(behavesdir+"behaves_"+depID+'.xlsx')
print("Loaded behaves file for... "+depID)
print(behaves.head())

# load in 1m tides file
onemintides = pd.read_csv(tidesdir+"tides\\tides_"+depID+'.csv')
print("Loaded 1m tides file for... "+depID)

# Look for summarizing tidal info file, if it doesn't exist, it needs to be
# created with the temp file in the other directory
try:
    sumtides = pd.read_excel(tidesdir+'sumtides\\sumtides_'+depID+'.xlsx')
    print("Loaded sumtides file for... "+depID)
    flag1 = 0
    print(sumtides.head())
except FileNotFoundError as e:
    print("No sumtides file found, loading sumtides-temp file...")
    sumtidesTEMP= pd.read_excel(tidesdir+'sumtides-temp\\sumtides-temp_'+depID+'.xlsx')
    print("Loaded sumtides-temp file for... "+depID)
    flag1 = 1
    print(sumtidesTEMP.head())

# ----------------------------------------- DATA PREP ---------------------------------- #
# # DO SOME MANIPULATIONS TO PRH, NAMELY, CREATE JERK AND SMOOTH IT, CONVERT
# # DATENUMS TO WORKABLE DATE TIMES, GET SOME TAG ON/OFF VARIABLES.

prh['roll'] = np.degrees(prh['roll'])
prh = make_jerk(prh)
prh = smooth_jerk(prh, method='gaussian', sigma=2)
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

# # ------------------------ TIDAL DATA PREP -----------------------------------#
# # IF IT DOESN'T EXIST ALREADY, CREATE A CSV WHICH STORES INFORMATION ABOUT
# # THE TIDES AT EACH PRH INDEX BASED ON 1 MIN TIDAL DATUM
try:
    os.path.exists(tidesdir+"prh-tides\\prh-tides_"+depID+".csv")
    print("Prh-tides file already created for "+depID+" adding to prh...")
    onemintides = pd.read_csv(tidesdir+"prh-tides\\prh-tides_"+depID+".csv")
    prh['tidalheight'] = onemintides['Prediction'].values
    prh['tidedirection'] = onemintides['direction'].values
    prh['tiderate'] = onemintides['rate'].values
except FileNotFoundError as e:
    sloc = tidesdir+"prh-tides\\prh-tides_"+depID+".csv"
    prh = add_tides(prh, onemintides, sloc)

# # IF YOU HAVEN'T ALREADY DONE IT, FINALIZE THE SUMMARIZED TIDES DATA,
# # REMOVE THE TEMP FILE
if flag1 == 1:
    sumtides = analyze_tides(prh, sumtidesTEMP, tonDT, toffDT)
    sumtides.to_excel(tidesdir+'sumtides\\sumtides_'+depID+'.xlsx', index=False)
    os.remove(tidesdir+'sumtides-temp\\sumtides-temp_'+depID+'.xlsx')
    print("Created sumtides file for... "+depID)
else:
    sumtides['endTD'] = pd.to_datetime(sumtides['endTD'])
    sumtides['durTD'] = pd.to_timedelta(sumtides['durTD'])
    print("No sumtides edits needed for... "+depID)

# #GENERATE A PLOT OF TIDES, ROLL, AND DEPTH FOR INSPECTION
saveloc = feeddir+"plots\\tide+depth+roll\\tides+depth+roll_"+depID+".png"
if not os.path.exists(saveloc):
    plot_tides(onemintides, sumtides, prh, tonI, toffI, saveloc)
else:
    print("Tide plot already made...")

# # ------------------------ FEED AUDIT -----------------------------------#

# LOOK FOR TEMP FEED FILE IF IT DOESN'T EXIST START FRESH.
try:
    feed = pd.DataFrame(pd.read_csv(feeddir+"accepted-feedevents\\accepted-feedevents_"+depID+".csv"))
    print("feed file loading...")
    print(feed.head())
    flag2 = 0
except FileNotFoundError as e:
    if not os.path.exists(feeddir+"progress-feedevents_"+depID+".csv"):
        print("No feeding file found. Starting fresh...")
        feed = pd.DataFrame(columns = ['sI', 'sFP', 'eFP', 'eI', 'jSig'])
        flag2 = 1

    else:
        print("Progress file found, loading...")
        feed = load_progressfile(feeddir+"progress-feedevents_"+depID+".csv")
        flag2 = 1

if flag2 == 1:
    feed = do_feedaudit(prh, feed, tonI, toffI, depID)
    pd.DataFrame(feed, columns=['sI', 'eI', 'sFP', 'eFP', 'jSig'])
    feed = check_jerkindices(feed, prh)
    feed = refine_jerkindices(feed, prh)
    feed = accept_events(feed, prh, depID)
    feed.to_csv(feeddir+"accepted-feedevents\\accepted-feedevents_"+depID+".csv", index=False)
    try:
        os.remove(feeddir + "progress-feedevents_" + depID + ".csv")
    except FileNotFoundError as e:
        print("No progress file to remove...")

else:
    print("Feeding audit already completed for... "+depID)

# # ----------------------------------- event analysis-------------+----------------------------- # #

try:
    feed = pd.DataFrame(pd.read_csv(feeddir+"analyzed-feedevents\\analyzed-feedevents_"+depID+".csv"))
    print("Analyzed feed file loading...")
    print(feed.head())
    #feed = check_dups(feed, prh)
    #feed.to_csv(feeddir + "analyzed-feedevents\\analyzed-feedevents_" + depID + ".csv")
    prh = get_rollratePRH(prh,'roll', fs = 10)
    fePRH = get_feedPRH(prh,feed)
    fePRH.to_csv(feeddir+"prh-feedevents\\prh-feedevents_"+depID+".csv", index=False)
except FileNotFoundError as e:
    print('No analyzed feed file found. Starting fresh...')
    feed = pd.DataFrame(pd.read_csv(feeddir + "accepted-feedevents\\accepted-feedevents_" + depID + ".csv"))
    print(feed.head())
    feed = analyze_feed(feed,prh, sumtides, depID)
    behaves = pd.DataFrame(behaves)
    behaves = behaves[behaves['State'] == 'feed']
    feed = add_behave(feed, behaves)
    feed = check_dups(feed, prh)
    feed = do_boutanalysis(feed)
    feed = get_PitCount(feed, prh)
    fePRH = get_feedPRH(prh,feed)
    feed.to_csv(feeddir + "analyzed-feedevents\\analyzed-feedevents_" + depID + ".csv")
    fePRH.to_csv(feeddir+"prh-feedevents\\prh-feedevents_"+depID+".csv", index=False)
    print('Analysis complete & saved...')

