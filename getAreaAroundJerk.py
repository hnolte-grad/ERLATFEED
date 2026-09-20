import pandas as pd
import numpy as np
import os
import glob

# ==== CONFIG ====
area_folder = r"C:\\_temp workspace\\1ER-LAT-FEED\\data\\3d model\\surface area\\area out"
events_csv = r"C:\\_temp workspace\\1ER-LAT-FEED\\data\\feed\\analyzed-feedevents\\analyzed-feedevents_MASTER.csv"
output_folder = r"C:\\_temp workspace\\1ER-LAT-FEED\\data\\3d model\\surface area\\area around jerk"
fs = 10  # Hz
window = fs  # ±1 sec = ±10 samples

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load events CSV
events_df = pd.read_csv(events_csv)

# Find all area files in folder
area_files = glob.glob(os.path.join(area_folder, "*-area.csv"))
print(area_files)
for area_file in area_files:
    # Get feedID by stripping "_area.csv"
    feedID = os.path.basename(area_file).replace("-area.csv", "")

    # Load area data
    area_df = pd.read_csv(area_file)

    # Find row in events
    row = events_df.loc[events_df['feedID'] == feedID]
    if row.empty:
        print(f"No matching event for {feedID}, skipping.")
        continue

    sI = int(row['sI'].values[0])
    jSig_raw = str(row['jSig'].values[0])

    # Convert jSig string into list of ints
    if jSig_raw.strip("[]").strip() == "":
        jSig_list = []
    else:
        jSig_list = [int(x.strip()) for x in jSig_raw.strip("[]").split(",") if x.strip().isdigit()]

    results = []
    for js in jSig_list:
        idx = sI - js

        # Get ±1 sec window
        start = max(0, idx - window)
        end = min(len(area_df), idx + window + 1)

        if 'area_m2' not in area_df.columns:
            raise ValueError(f"'area_m2' column not found in {area_file}")

        values = area_df['area_m2'].iloc[start:end]
        mean_area = values.mean() if len(values) > 0 else np.nan

        results.append({"jSig": js, "mean_area_m2": mean_area})

    # Save results
    out_df = pd.DataFrame(results)
    out_file = os.path.join(output_folder, f"{feedID}_mean area jerk.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Saved {out_file}")

# Store all results for summary CSV
all_results = []

for area_file in area_files:
    # Get feedID by stripping "_area.csv"
    feedID = os.path.basename(area_file).replace("_area.csv", "")

    # Load area data
    area_df = pd.read_csv(area_file)

    # Find row in events
    row = events_df.loc[events_df['feedID'] == feedID]
    if row.empty:
        print(f"No matching event for {feedID}, skipping.")
        continue

    sI = int(row['sI'].values[0])
    jSig_raw = str(row['jSig'].values[0])

    # Convert jSig string into list of ints
    if jSig_raw.strip("[]").strip() == "":
        jSig_list = []
    else:
        jSig_list = [int(x.strip()) for x in jSig_raw.strip("[]").split(",") if x.strip().isdigit()]

    results = []
    for js in jSig_list:
        # Shift from global index space → local area file index space
        idx = js - sI

        # Get ±1 sec window around local index
        start = max(0, idx - window)
        end = min(len(area_df), idx + window + 1)

        if 'area_m2' not in area_df.columns:
            raise ValueError(f"'area_m2' column not found in {area_file}")

        values = area_df['area_m2'].iloc[start:end]
        mean_area = values.mean() if len(values) > 0 else np.nan

        results.append({"feedID": feedID, "jSig": js, "mean_area_m2": mean_area})

    # Save individual file
    out_df = pd.DataFrame(results)
    out_file = os.path.join(output_folder, f"{feedID}_mean area jerk.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Saved {out_file}")

    # Append to master list
    all_results.append(out_df)

# Compile everything into one CSV
if all_results:
    summary_df = pd.concat(all_results, ignore_index=True)
    summary_file = os.path.join(output_folder, "all_mean_area_jerk.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\nCompiled all results into {summary_file}")
else:
    print("\nNo results to compile.")
