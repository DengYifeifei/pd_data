import os
import pandas as pd
import numpy as np
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import DataAnalyze_new

# --- master list of folders to process ---
included_folders = [
    # '25-10-07-1233_setting0', 
    # '25-10-07-1328_setting0', 
    # '25-10-09-1303_setting0', 
    # '25-10-27-1254_setting0', 
    # '25-10-31-1350_setting0', 
    '25-11-03-1100_setting0',
    '25-11-04-1214_setting0'
]

excluded_folders = [
    # any folders you want to skip
]
# included_folders = [
#     # '25-11-14-1255_setting0',
#     '25-11-11-1259_setting0',
#     '25-11-14-1455_setting0',
#     '25-11-14-1354_setting0',
#     '25-11-18-1258_setting0',
#     '25-11-11-1358_setting0',
#     '25-11-18-1456_setting0',
#     '25-11-13-1307_setting0',
#     '25-11-18-1213_setting0',
#     '25-11-13-1214_setting0'
# ]
# included_folders = [
#     # '25-10-17-1206_setting0',
#     # '25-10-24-1302_setting0',
#     # '25-10-30-1254_setting0',
#     # '25-10-22-1358_setting0',
#     # '25-09-30-1530_setting0',
#     # '25-09-30-1338_setting0',
#     # '25-10-01-1417_setting0',
#     # '25-11-05-1457_setting0',
#     # '25-09-29-1238_setting0',
#     # '25-11-03-1151_setting0',
#     # '25-11-04-1404_setting0',
#     '25-11-07-1456_setting0',
#     # '25-11-05-1407_setting0',
#     # '25-09-30-1427_setting0',
#     # '25-10-08-1408_setting0',
#     # '25-10-20-1204_setting0',
#     '25-10-07-1513_setting0',
#     # '25-10-21-1446_setting0',
#     '25-11-05-1407_setting0',
#     # '25-10-23-1258_setting0',
#     # '25-09-30-1142_setting0',
#     # '25-10-24-1354_setting0',
#     # '25-10-20-1257_setting0',
#     # '25-11-03-1301_setting0',
#     '25-11-07-1401_setting0',
#     '25-11-06-1302_setting0'
# ]

# excluded_folders = [
#     '25-10-24-1302_setting0',
#     '25-10-30-1254_setting0',
#     '25-10-22-1358_setting0',
#     '25-09-30-1530_setting0',
#     '25-10-01-1417_setting0',
#     '25-09-30-1427_setting0',
#     '25-10-20-1204_setting0',
#     '25-10-07-1513_setting0',
#     '25-10-21-1446_setting0',
#     '25-10-27-1205_setting0'
# ]
version = 'v1'
folders_to_process = [folder for folder in included_folders if folder not in excluded_folders]
base_directory = f"data/processed/{version}"

# --- loop through each folder ---
for session_name in folders_to_process:
    print(f"\n🔹 Processing folder: {session_name}")
    foldername = os.path.join(base_directory, "eyetracking", session_name)

    # load pilot
    pilot = DataAnalyze_new(session_name, base_directory)

    # --- combine baseline blocks ---
    baseline_files = sorted([f for f in os.listdir(foldername) if f.startswith('baseline_block') and f.endswith('.csv')])
    if not baseline_files:
        print(f"⚠️ No baseline_block CSVs found for {session_name}, skipping.")
        continue
    baseline_combined = pd.concat([pd.read_csv(os.path.join(foldername, f)) for f in baseline_files], ignore_index=True)
    baseline_combined.to_csv(os.path.join(foldername, 'baseline.csv'), index=False)
    print("✅ baseline blocks combined")

    # --- combine filtered_eye blocks ---
    filtered_files = sorted([f for f in os.listdir(foldername) if f.startswith('filtered_eye_block') and f.endswith('.csv')])
    if not filtered_files:
        print(f"⚠️ No filtered_eye_block CSVs found for {session_name}, skipping.")
        continue
    filtered_combined = pd.concat([pd.read_csv(os.path.join(foldername, f)) for f in filtered_files], ignore_index=True)
    filtered_combined.to_csv(os.path.join(foldername, 'filtered_eye.csv'), index=False)
    print("✅ filtered_eye blocks combined")

    # # --- check for NaNs in baseline ---
    # if baseline_combined['baseline'].isnull().any():
    #     print(f"❌ baseline column contains NaN in {session_name}, skipping.")
    #     continue

    # --- compute sound-locked pupil stats ---
    baseline_df = baseline_combined.copy()
    filtered_eye = filtered_combined.copy()

    sound_events = pilot.eye_raw[pilot.eye_raw['Event'] == 'start sound'][['trial_index', 'TimeEvent']]
    sound_events = sound_events[sound_events['trial_index'].isin(baseline_df['trial_index'])].reset_index(drop=True)

    baseline_df['max_pupilSound'] = np.nan
    baseline_df['mean_pupilSound'] = np.nan
    baseline_df['median_pupilSound'] = np.nan

    for row in sound_events.itertuples():
        t = row.TimeEvent
        tri = row.trial_index

        mask = (
            (filtered_eye['trial_index'] == tri) &
            (filtered_eye['TimeEvent'] >= t + 0.5) &
            (filtered_eye['TimeEvent'] <= t + 2.0)
        )
        window = filtered_eye.loc[mask, 'Pupil']

        if not window.empty:
            baseline_df.loc[baseline_df['trial_index'] == tri, 'max_pupilSound'] = window.max()
            baseline_df.loc[baseline_df['trial_index'] == tri, 'mean_pupilSound'] = window.mean()
            baseline_df.loc[baseline_df['trial_index'] == tri, 'median_pupilSound'] = window.median()
        else:
            print(f"⚠️ No pupil data for trial_index {tri}")

    # --- save final DataFrame ---
    out_csv = os.path.join(foldername, 'baseline_with_pupilSound.csv')
    baseline_df.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}")

    # --- delete old block files ---
    for f in baseline_files + filtered_files:
        os.remove(os.path.join(foldername, f))
    print(f"🗑️ Removed old block files for {session_name}")

print("🎉 All done! Processed all included folders.")
