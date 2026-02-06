import os
import numpy as np
from multiprocessing import Pool
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import DataAnalyze_new  

version = 'v1'
base_directory = f"data/processed/{version}"
eye_directory = os.path.join(base_directory, "eyetracking")

# excluded_folders = ["25-03-07-1109_setting0"]  # Add more if needed
# ------------------------
# Folder inclusion/exclusion
# ------------------------
included_folders = [
    '25-10-07-1233_setting0', 
    '25-10-07-1328_setting0', 
    '25-10-09-1303_setting0', 
    '25-10-27-1254_setting0', 
    '25-10-31-1350_setting0', 
    '25-11-03-1100_setting0',
    '25-11-04-1214_setting0'
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

excluded_folders = []
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

def process_folder(folder):
    # folder_path = os.path.join(base_directory, folder)

    # if folder == ".DS_Store" or folder in excluded_folders:
    #     return

    if folder not in included_folders or folder in excluded_folders:
        print(f"⏭️ Skipping folder {folder}")
        return

    # Check if the file already exists in that full path
    raw_pupil_file = os.path.join(eye_directory, folder, "rawPupil_sound.npy")
    if os.path.exists(raw_pupil_file):
        print(f"Skipping folder {folder} as rawPupil_sound.npy already exists.")
        return
    
    print(f"Starting folder {folder}")
    pilot = DataAnalyze_new(folder, f"data/processed/{version}")
    pupils_alltrials = []

    for i in range(1, 271):
        # try:
        #     if i == 1:
        #         trials = [i]
        #     else:
        #         trials = [i-1, i]

        #     neighbor_trial_data = pilot.eye_raw[pilot.eye_raw['trial_index'].isin(trials)].copy()
        #     message_data = neighbor_trial_data[neighbor_trial_data['Type'] == 'Message']
        #     pupil_data = pilot.filter_eye_data_diff_blink(neighbor_trial_data, trials)

        #     # if i == 1:
        #     cue_time = message_data[(message_data['trial_index'] == i) & 
        #                                 (message_data['Event'] == 'show cue')]['TimeEvent'].iloc[0]
   
        #         # dt = nextCue_time - sound_time
        #     # else:
        #     #     sound_time = message_data[message_data['Event'] == 'start sound']['TimeEvent'].iloc[0]
        #     #     dt = 10

        #     # if dt <= 5:
        #     filtered_pupil_data = pupil_data[
        #         (pupil_data['TimeEvent'] >= cue_time - 1) & 
        #         (pupil_data['TimeEvent'] <= cue_time + 2)
        #     ]

        #     time_data = filtered_pupil_data['TimeEvent'].copy()
        #     rounded_indices = np.ceil((time_data - (cue_time - 1)) / 0.001).astype(int)

        #     duplicates = rounded_indices.value_counts()
        #     rounded_indices = rounded_indices[~rounded_indices.duplicated(keep='first')]

        #     pupils_1trial = np.full(3000, np.nan)
        #     for idx, rounded_idx in rounded_indices.items():
        #         pupil_size = filtered_pupil_data.loc[idx, 'Pupil']
        #         if 0 < rounded_idx <= 3000:
        #             pupils_1trial[rounded_idx - 1] = pupil_size

        #     pupils_alltrials.append(pupils_1trial)
        #     print(f"Done with trial {i} in file {folder}")
        # except Exception as e:
        #     print(f"Error in trial {i}, folder {folder}: {e}")
        #     continue


        try:
            if i < 270:
                trials = [i, i + 1]
            else:
                trials = [i]

            neighbor_trial_data = pilot.eye_raw[pilot.eye_raw['trial_index'].isin(trials)].copy()
            message_data = neighbor_trial_data[neighbor_trial_data['Type'] == 'Message']
            pupil_data = pilot.filter_eye_data_diff_blink(neighbor_trial_data, trials)

            if i < 270:
                sound_time = message_data[(message_data['trial_index'] == i) & 
                                          (message_data['Event'] == 'start sound')]['TimeEvent'].iloc[0]
                nextCue_time = message_data[(message_data['trial_index'] == i + 1) & 
                                            (message_data['Event'] == 'show cue')]['TimeEvent'].iloc[0]
                dt = nextCue_time - sound_time
            else:
                sound_time = message_data[message_data['Event'] == 'start sound']['TimeEvent'].iloc[0]
                dt = 10

            if dt <= 5:
                filtered_pupil_data = pupil_data[
                    (pupil_data['TimeEvent'] >= sound_time - 1) & 
                    (pupil_data['TimeEvent'] <= nextCue_time)
                ]
            else:
                filtered_pupil_data = pupil_data[
                    (pupil_data['TimeEvent'] >= sound_time - 1) & 
                    (pupil_data['TimeEvent'] <= sound_time + 5)
                ]

            time_data = filtered_pupil_data['TimeEvent'].copy()
            rounded_indices = np.ceil((time_data - (sound_time - 1)) / 0.001).astype(int)

            duplicates = rounded_indices.value_counts()
            rounded_indices = rounded_indices[~rounded_indices.duplicated(keep='first')]

            pupils_1trial = np.full(6000, np.nan)
            for idx, rounded_idx in rounded_indices.items():
                pupil_size = filtered_pupil_data.loc[idx, 'Pupil']
                if 0 < rounded_idx <= 6000:
                    pupils_1trial[rounded_idx - 1] = pupil_size

            pupils_alltrials.append(pupils_1trial)
            print(f"Done with trial {i} in file {folder}")
        except Exception as e:
            print(f"Error in trial {i}, folder {folder}: {e}")
            continue

    save_path = os.path.join(eye_directory, folder, "rawPupil_sound.npy")
    np.save(save_path, np.array(pupils_alltrials, dtype=object))
    print(f"Saved data for folder {folder}")

if __name__ == "__main__":
    # folders = [f for f in os.listdir(eye_directory) 
    #            if f not in excluded_folders and f != ".DS_Store"]
    
    folders = [f for f in os.listdir(eye_directory) if f in included_folders]

    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_folder, folders)
