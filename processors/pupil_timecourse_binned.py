import os
import numpy as np
from multiprocessing import Pool
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import DataAnalyze_new  

# -------------------------------
# Config
# -------------------------------
version = 'v1'
base_directory = f"data/processed/{version}"
eye_directory = os.path.join(base_directory, "eyetracking")

yes_folders = [
    '25-09-29-1238_setting0',
    '25-09-30-1142_setting0',
    '25-09-30-1338_setting0',
    '25-09-30-1427_setting0',
    '25-09-30-1530_setting0',
    '25-10-01-1417_setting0',
    '25-10-07-1233_setting0',
    '25-10-07-1328_setting0',
    '25-10-07-1513_setting0',
    '25-10-09-1303_setting0',
    '25-10-17-1206_setting0',
    '25-10-20-1204_setting0',
    '25-10-20-1257_setting0',
    '25-10-21-1446_setting0',
    '25-10-22-1358_setting0',
    '25-10-23-1258_setting0',
    '25-10-24-1302_setting0',
    '25-10-27-1254_setting0',
    '25-10-30-1254_setting0',
    '25-10-31-1350_setting0',
    '25-11-03-1100_setting0',
    '25-11-03-1151_setting0',
    '25-11-03-1301_setting0',
    '25-11-04-1214_setting0',
    '25-11-04-1404_setting0',
    '25-11-05-1407_setting0',
    '25-11-05-1457_setting0',
    '25-11-06-1302_setting0',
    '25-11-07-1401_setting0',
    '25-11-07-1456_setting0'
]

trial_index_list = np.arange(1, 271)
bin_size = 0.1  # seconds


# -------------------------------
# Per-folder processing function
# -------------------------------
def process_folder(folder):

    save_path = os.path.join(
        eye_directory, folder, "pupil_timecourse_binned.npy"
    )

    # ---- Skip if already processed ----
    if os.path.exists(save_path):
        print(f"[SKIP] {folder}: file already exists")
        return

    print(f"[PROCESS] {folder}")

    participant = DataAnalyze_new(folder, base_directory)

    pupil_data_clean = participant.filter_eye_data_diff_blink(
        participant.eye_raw, trial_index_list
    )

    pupil_timecourse_binned = []  # list of arrays, one per trial

    # -------------------------------
    # Loop over trials
    # -------------------------------
    for i in trial_index_list:

        trial_eye_data = pupil_data_clean[
            pupil_data_clean['trial_index'] == i
        ]

        t_start_vals = participant.eye_raw[
            (participant.eye_raw['Type'] == 'Message') &
            (participant.eye_raw['trial_index'] == i) &
            (participant.eye_raw['Event'] == 'initialize')
        ]['TimeEvent'].values

        if len(t_start_vals) == 0 or trial_eye_data.empty:
            pupil_timecourse_binned.append(np.array([]))
            continue

        t_start = t_start_vals[0]

        # -------------------------------
        # Extract and clean data
        # -------------------------------
        times = trial_eye_data['TimeEvent'].values - t_start
        pupils = trial_eye_data['Pupil'].values

        valid = (~np.isnan(times)) & (~np.isnan(pupils))
        times = times[valid]
        pupils = pupils[valid]

        if times.size == 0:
            pupil_timecourse_binned.append(np.array([]))
            continue

        # -------------------------------
        # Bin
        # -------------------------------
        max_time = times.max()
        bin_edges = np.arange(0, max_time + bin_size, bin_size)

        bin_idx = np.digitize(times, bin_edges) - 1

        bin_means = np.array([
            np.nanmean(pupils[bin_idx == b]) if np.any(bin_idx == b) else np.nan
            for b in range(len(bin_edges) - 1)
        ])

        pupil_timecourse_binned.append(bin_means)

    # -------------------------------
    # Save once per folder
    # -------------------------------
    np.save(
        save_path,
        np.array(pupil_timecourse_binned, dtype=object),
        allow_pickle=True
    )
    # pupil_timecourse_binned = np.load(save_path, allow_pickle=True)
    print(f"[SAVED] {folder}")


# -------------------------------
# Multiprocessing entry point
# -------------------------------
if __name__ == "__main__":

    folders = [
        f for f in os.listdir(eye_directory)
        if f in yes_folders
    ]

    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_folder, folders)
