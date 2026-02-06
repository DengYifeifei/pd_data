import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import DataAnalyze_new  # adjust sys.path if needed

# ------------------------
# Directories
# ------------------------
version = 'v1'
eye_directory = f'data/processed/{version}/eyetracking'
base_directory = f'data/processed/{version}'

included_folders = [
    # '25-10-07-1233_setting0', 
    # '25-10-07-1328_setting0', 
    # '25-10-09-1303_setting0', 
    # '25-10-27-1254_setting0', 
    # '25-10-31-1350_setting0', 
    '25-11-03-1100_setting0'
    # '25-11-04-1214_setting0'
]

excluded_folders = [
    # any folders you want to skip
]

# ------------------------
# Process a single folder
# ------------------------
def process_folder(folder):
    folder_path = os.path.join(eye_directory, folder)
    os.makedirs(folder_path, exist_ok=True)

    print(f"🔵 Starting folder {folder}")
    pilot = DataAnalyze_new(folder, base_directory)

    for block_number in range(1, 10):  # Blocks 1 to 9
        baseline_file = os.path.join(folder_path, f'baseline_block{block_number}.csv')
        filtered_eye_file = os.path.join(folder_path, f'filtered_eye_block{block_number}.csv')

        if os.path.exists(baseline_file) and os.path.exists(filtered_eye_file):
            print(f"✅ Block {block_number} already exists, skipping.")
            continue

        start_trial = (block_number - 1) * 30 + 1
        end_trial = block_number * 30 + 1  # exclusive
        trial_list = list(range(start_trial, end_trial))

        print(f"🔸 Processing Block {block_number}: Trials {start_trial} to {end_trial - 1}")

        # Filter eye data
        filtered_eye = pilot.filter_eye_data_diff_blink(pilot.eye_raw, trial_list)

        # Compute baseline for each trial
        baseline = []
        for trial in trial_list:
            cue_time_row = pilot.eye_raw.loc[
                (pilot.eye_raw['Event'] == 'show cue') &
                (pilot.eye_raw['trial_index'] == trial),
                'TimeEvent'
            ]
            if not cue_time_row.empty:
                t = cue_time_row.values[0]
                mask = (filtered_eye['TimeEvent'] >= t - 0.5) & (filtered_eye['TimeEvent'] <= t)
                window_data = filtered_eye.loc[mask, 'Pupil']
                baseline.append(window_data.mean() if not window_data.empty else np.nan)
            else:
                print(f"⚠️ Baseline unavailable for trial {trial}")
                baseline.append(np.nan)

        # Ensure lengths match
        assert len(trial_list) == len(baseline), f"Length mismatch in folder {folder}, block {block_number}"

        # Save baseline and filtered eye data
        baseline_df = pd.DataFrame({
            'trial_index': trial_list,
            'baseline': baseline
        })
        baseline_df.to_csv(baseline_file, index=False)
        filtered_eye.to_csv(filtered_eye_file, index=False)

        print(f"✅ Block {block_number} done and saved.")

    print(f"✅ Finished folder {folder}.\n")


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    all_folders = [f for f in os.listdir(eye_directory) if f not in ['.DS_Store']]
    folders_to_process = [f for f in all_folders if f in included_folders and f not in excluded_folders]

    print(f"🔹 Folders to process: {folders_to_process}\n")

    # Use multiprocessing
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_folder, folders_to_process)

    print("🎉 All folders processed.")

# import os
# import pandas as pd
# import numpy as np
# from multiprocessing import Pool

# import sys
# # Add the parent directory to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# from functions import DataAnalyze_new  

# # ------------------------
# # Directories
# # ------------------------
# version = 'v1'
# base_directory = f"data/processed/{version}"
# eye_directory = os.path.join(base_directory, "eyetracking")

# # ------------------------
# # Folder inclusion/exclusion
# # ------------------------
# # included_folders = [
# #     # '25-11-14-1255_setting0',
# #     '25-11-11-1259_setting0',
# #     '25-11-14-1455_setting0',
# #     '25-11-14-1354_setting0',
# #     '25-11-18-1258_setting0',
# #     '25-11-11-1358_setting0',
# #     '25-11-18-1456_setting0',
# #     '25-11-13-1307_setting0',
# #     '25-11-18-1213_setting0',
# #     '25-11-13-1214_setting0'
# # ]
# included_folders = [
#     # '25-10-07-1233_setting0', 
#     # '25-10-07-1328_setting0', 
#     # '25-10-09-1303_setting0', 
#     # '25-10-27-1254_setting0', 
#     # '25-10-31-1350_setting0', 
#     '25-11-03-1100_setting0', 
#     # '25-11-04-1214_setting0'
# ]
# # included_folders = [
# #     # '25-10-17-1206_setting0',
# #     # '25-10-24-1302_setting0',
# #     # '25-10-30-1254_setting0',
# #     # '25-10-22-1358_setting0',
# #     # '25-09-30-1530_setting0',
# #     # '25-09-30-1338_setting0',
# #     # '25-10-01-1417_setting0',
# #     # '25-11-05-1457_setting0',
# #     # '25-09-29-1238_setting0',
# #     # '25-11-03-1151_setting0',
# #     # '25-11-04-1404_setting0',
# #     '25-11-07-1456_setting0',
# #     # '25-11-05-1407_setting0',
# #     # '25-09-30-1427_setting0',
# #     # '25-10-08-1408_setting0',
# #     # '25-10-20-1204_setting0',
# #     '25-10-07-1513_setting0',
# #     # '25-10-21-1446_setting0',
# #     '25-11-05-1407_setting0',
# #     # '25-10-23-1258_setting0',
# #     # '25-09-30-1142_setting0',
# #     # '25-10-24-1354_setting0',
# #     # '25-10-20-1257_setting0',
# #     # '25-11-03-1301_setting0',
# #     '25-11-07-1401_setting0',
# #     '25-11-06-1302_setting0'
# # ]
# excluded_folders = [] 
# # excluded_folders = [
# #     '25-10-24-1302_setting0',
# #     '25-10-30-1254_setting0',
# #     '25-10-22-1358_setting0',
# #     '25-09-30-1530_setting0',
# #     '25-10-01-1417_setting0',
# #     '25-09-30-1427_setting0',
# #     '25-10-20-1204_setting0',
# #     '25-10-07-1513_setting0',
# #     '25-10-21-1446_setting0',
# #     '25-10-27-1205_setting0'
# # ]

# # ------------------------
# # Processing function
# # ------------------------
# def process_folder(folder):

#     # if folder not in included_folders or folder in excluded_folders:
#     #     print(f"⏭️ Skipping folder {folder}")
#     #     return

#     folder_path = os.path.join(eye_directory, folder)
#     os.makedirs(folder_path, exist_ok=True)

#     print(f"🔵 Starting folder {folder}")
#     pilot = DataAnalyze_new(folder, base_directory)

#     for block_number in range(1, 10):  # Blocks 1 to 9
#         baseline_file = os.path.join(folder_path, f'baseline_block{block_number}.csv')
#         filtered_eye_file = os.path.join(folder_path, f'filtered_eye_block{block_number}.csv')

#         if os.path.exists(baseline_file) and os.path.exists(filtered_eye_file):
#             print(f"✅ Block {block_number} already exists, skipping.")
#             continue

#         start_trial = (block_number - 1) * 30 + 1
#         end_trial = block_number * 30 + 1  # Exclusive
#         trial_list = list(range(start_trial, end_trial))

#         print(f"🔸 Processing Block {block_number}: Trials {start_trial} to {end_trial - 1}")

#         cue_times = pilot.eye_raw.loc[
#             (pilot.eye_raw['Event'] == 'show cue') &
#             (pilot.eye_raw['trial_index'].isin(trial_list)),
#             'TimeEvent'
#         ]

#         filtered_eye = pilot.filter_eye_data_diff_blink(pilot.eye_raw, trial_list)

#         baseline = []
#         for t in cue_times:
#             mask = (filtered_eye['TimeEvent'] >= t - 0.5) & (filtered_eye['TimeEvent'] <= t)
#             window_data = filtered_eye.loc[mask, 'Pupil']

#             if not window_data.empty:
#                 baseline.append(window_data.mean())
#             else:
#                 print(f"⚠️ Baseline unavailable for cue time {t}")
#                 baseline.append(np.nan)

#         baseline_df = pd.DataFrame({
#             'trial_index': range(start_trial, end_trial),
#             'baseline': baseline
#         })

#         baseline_df.to_csv(baseline_file, index=False)
#         filtered_eye.to_csv(filtered_eye_file, index=False)

#         print(f"✅ Block {block_number} done and saved.")

#     print(f"✅ Finished folder {folder}.\n")


# # ------------------------
# # Main
# # ------------------------
# if __name__ == "__main__":
#     all_folders = [f for f in os.listdir(eye_directory) 
#                    if f not in ['.DS_Store']]

#     folders_to_process = [f for f in all_folders 
#                           if f in included_folders and f not in excluded_folders]

#     print(f"🔹 Folders to process: {folders_to_process}\n")

#     with Pool(processes=os.cpu_count()) as pool:
#         pool.map(process_folder, folders_to_process)

#     print("🎉 All folders processed.")
