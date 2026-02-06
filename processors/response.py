import os
import sys
import numpy as np
from multiprocessing import Pool

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import DataAnalyze_new

version = "v1"
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
    '25-11-07-1456_setting0',
]

def process_folder(folder):
    raw_pupil_file = os.path.join(eye_directory, folder, "rawPupil_response.npy")
    # if os.path.exists(raw_pupil_file):
    #     print(f"⏭️ Skipping {folder} (already processed)")
    #     return

    print(f"▶️ Starting folder {folder}")
    pilot = DataAnalyze_new(folder, base_directory)

    pupils_alltrials = []

    for i in range(1, 271):
        try:
            trial_data = pilot.eye_raw[
                pilot.eye_raw["trial_index"] == i
            ].copy()

            response_t = trial_data.loc[
                (trial_data["Type"] == "Message") &
                (trial_data["Event"] == "response"),
                "TimeEvent"
            ].values

            trial_end_t = trial_data.loc[
                (trial_data["Type"] == "Message") &
                (trial_data["Event"] == "done"),
                "TimeEvent"
            ].values

            # Skip if events are missing
            if len(response_t) == 0 or len(trial_end_t) == 0:
                pupils_alltrials.append(None)
                continue

            response_t = response_t[0]
            trial_end_t = trial_end_t[0]

            pupil_data_filtered = pilot.filter_eye_data_diff_blink(
                trial_data, [i]
            )

            pupil_after_response = pupil_data_filtered[
                (pupil_data_filtered["TimeEvent"] > response_t+0.5) &
                (pupil_data_filtered["TimeEvent"] < min(trial_end_t, response_t+2))
            ]

            pupils_alltrials.append(pupil_after_response)

        except Exception as e:
            print(f"❌ Error in trial {i}, folder {folder}: {e}")
            pupils_alltrials.append(None)

    save_path = os.path.join(
        eye_directory, folder, "rawPupil_response.npy"
    )
    np.save(save_path, np.array(pupils_alltrials, dtype=object))
    print(f"✅ Saved data for folder {folder}")


if __name__ == "__main__":
    folders = [
        f for f in os.listdir(eye_directory)
        if f in yes_folders
    ]

    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_folder, folders)
