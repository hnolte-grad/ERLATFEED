################################## import functions ###################################

from workhorseFuncs import *
import warnings
from scipy.io.matlab import _mio5
import glob
from scipy.stats import circmean, circstd  # <-- added
warnings.filterwarnings("ignore", category=_mio5.MatReadWarning)

##################################### MAIN  ########################################
#----------------------LOAD IN FILES, SET THE ENVIRONMENT -------------------------#
wdir = "C:\\_temp workspace\\1ER-LAT-FEED\\data"
os.chdir(wdir)

prhdir = wdir + '\\tags\\prhs\\'
sdir = wdir + '\\tags\\speed\\'
bdir = wdir + '\\tags\\behaves\\'

prhfiles = os.listdir(prhdir)
#fdat = pd.read_csv(wdir+'\\feed\\analyzed-feedevents\\analyzed-feedevents_MASTER.csv')
bdat = pd.read_excel(bdir+'behaves_MASTERFEED.xlsx')

def addSpdPHeadtoBehaves(df, prh_dir, speed_dir, output_path):
    # Prepare output columns
    df['meanSpeed'] = np.nan
    df['meanHeading'] = np.nan       # circular mean of heading (rad)
    df['stdHeading'] = np.nan        # circular std of heading (rad)
    df['meanDepth'] = np.nan
    df['stdDepth'] = np.nan

    # Group by depID
    grouped = df.groupby('depID')

    for depID, group_df in grouped:
        print(f"Processing deployment: {depID}")

        # === Load speed file (CSV with JJ column) ===
        speed_file = os.path.join(speed_dir, f"{depID}_speed.csv")
        if not os.path.exists(speed_file):
            print(f"Missing speed file for {depID}, skipping.")
            continue
        sdat = pd.read_csv(speed_file)

        # === Load PRH .mat file to get heading and depth ===
        prh_file = os.path.join(prh_dir, f"{depID} 10Hzprh.mat")
        if not os.path.exists(prh_file):
            print(f"Missing PRH file for {depID}, skipping.")
            continue
        prh = sio.loadmat(prh_file, squeeze_me=True, struct_as_record=False)
        heading = np.array(prh.get('head', []))   # assumed radians, range ~[-pi, pi]
        depth = np.array(prh.get('p', []))

        if heading.size == 0 or depth.size == 0:
            print(f"Missing heading or depth in {depID}, skipping.")
            continue

        for idx, row in group_df.iterrows():
            sI = int(row['sInd'])
            eI = int(row['eInd'])

            # Check bounds
            if eI >= len(sdat) or eI >= len(heading) or eI >= len(depth):
                print(f"Index out of bounds for {depID} at row {idx}, skipping.")
                continue

            # Slice ranges
            speed_seg = sdat[sI:eI + 1]
            head_seg = heading[sI:eI + 1]
            depth_seg = depth[sI:eI + 1]

            # Mean speed (linear)
            df.at[idx, 'meanSpeed'] = np.mean(speed_seg)

            # Heading circular mean and std (in radians)
            df.at[idx, 'meanHeading'] = circmean(head_seg, high=np.pi, low=-np.pi)
            df.at[idx, 'stdHeading'] = circstd(head_seg, high=np.pi, low=-np.pi)

            # Depth mean and std (linear)
            df.at[idx, 'meanDepth'] = np.mean(depth_seg)
            df.at[idx, 'stdDepth'] = np.std(depth_seg)

    # Save to output Excel
    df.to_excel(output_path + "behaves_MASTERFEED.xlsx", index=False)
    print(f"Saved results to: {output_path}")

addSpdPHeadtoBehaves(bdat, prhdir, sdir, bdir)
