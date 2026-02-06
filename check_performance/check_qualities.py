import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def fetch_log(file_name, directory):
    """
    Load a JSON log file as a DataFrame.

    Args:
        file_name (str): Name of the log file without extension.
        directory (str): Base directory containing 'trial_data'.

    Returns:
        pd.DataFrame or None: Loaded DataFrame, or None if error occurs.
    """
    complete_path = os.path.join(directory, "trial_data", f"{file_name}.json")
    
    try:
        with open(complete_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: The file '{complete_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{complete_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

def add_block_columns(log_data, block_len=30):
    # Create a new column for the block number
    log_data['block_index'] = (log_data['trial_index'] - 1) // block_len + 1

    # Create a new column for trial index within each block
    log_data['trial_index_within_block'] = log_data['trial_index'] % block_len
    log_data['trial_index_within_block'] = log_data['trial_index_within_block'].replace(0, block_len)  # Ensure 0 becomes block_len

    return log_data

class CheckQuality:
    """
    Class to load and handle trial and eye-tracking data for quality checks.
    """
    def __init__(self, file_name):
        self.directory = "data/processed/v1"
        self.file_name = file_name

        # Paths
        self.log_path = os.path.join(self.directory, "trial_data", f"{file_name}.json")
        self.eye_data_path = os.path.join(self.directory, "eyetracking", file_name, "all_adin.csv")

        # Load eye-tracking data
        if os.path.exists(self.eye_data_path):
            self.eye_raw = pd.read_csv(self.eye_data_path)
        
        # Load trial log data
        self.log_data = fetch_log(file_name, self.directory)
        self.log_data = add_block_columns(self.log_data)

        self.incorrect_count = (self.log_data["performance"] == "Incorrect").sum()
        self.rt_mean = self.log_data["rt"].mean()
        
        self.num_blocks = 9
        self.trial_parameters = []
        for i in range(self.num_blocks):
            idx = i * 30
            block_sample = self.log_data.iloc[idx]
            left_cue_conditions = block_sample['left_cue_condition']
            stimA_conditions = block_sample['stimA_condition']
            self.trial_parameters.append((left_cue_conditions, stimA_conditions))

            
 

    def show_performance(self):
        print("file name:", self.file_name)

        """Plot incorrect trials per block and RT distribution."""
        # Total incorrect trials
        self.incorrect_trial_list = self.log_data.loc[self.log_data["performance"] == "Incorrect", "trial_index"]

        print("Incorrect count:", self.incorrect_count)
        print("Incorrect trial list:", self.incorrect_trial_list.values)
        print("Number of incorrect trials per block:")

        incorrect_trials_per_block = []
        block_labels = []

        for i in range(self.num_blocks):
            block_start = i * 30
            block_end   = (i + 1) * 30

            # Count incorrect trials in this block
            count_in_block = self.incorrect_trial_list.between(block_start, block_end - 1).sum()
            incorrect_trials_per_block.append(count_in_block)

            # Convert numpy floats to python floats for clean labels
            param = tuple(round(float(x), 3) for x in self.trial_parameters[i])
            block_labels.append(f"{param}, {count_in_block}")

            # Print original info
            print(f"{param}, {count_in_block} incorrect trials")

        # # Store results
        # self.incorrect_trials_per_block = incorrect_trials_per_block

        # # --- Bar plot of incorrect trials per block ---
        # plt.figure(figsize=(3, 2))
        # plt.bar(range(self.num_blocks), incorrect_trials_per_block)
        # plt.xticks(range(self.num_blocks), block_labels, rotation=45, ha='right')
        # plt.xlabel("Block (parameters, count)")
        # plt.ylabel("# Incorrect Trials")
        # plt.title("Incorrect Trials per Block")
        # plt.tight_layout()
        # plt.show()

        # --- RT histogram ---
        plt.figure(figsize=(2, 1))
        bins = np.arange(0, self.log_data["rt"].max() + 0.5, 0.5)
        plt.hist(self.log_data["rt"], bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel("RT (seconds)")
        plt.ylabel("Frequency")
        plt.title("RT Distribution")
        plt.grid(axis='y', linestyle="--", alpha=0.7)
        plt.show()


    def plot_pupil_per_block(self, fig_size=(40, 5), dot_size=5):
        """
        Plot pupil size against TimeEvent for each block.
        Blocks are defined by the first trial in each block.
        """
        if self.eye_raw.empty:
            print("No eye-tracking data available.")
            return

        first_trials = self.log_data[self.log_data['trial_index_within_block'] == 1]
        block_starts = first_trials['events'].apply(lambda ev: math.floor(ev[0]['time']))
        block_starts = block_starts.sort_values().tolist()
        block_starts.append(self.eye_raw['TimeEvent'].max() + 1)  # sentinel

        for i in range(len(block_starts) - 1):
            print("file name:", self.file_name)
            start = block_starts[i]
            end   = block_starts[i + 1]

            block_df = self.eye_raw[(self.eye_raw['TimeEvent'] >= start) & (self.eye_raw['TimeEvent'] < end)]
            block_df = block_df.dropna(subset=['Pupil', 'TimeEvent'])

            plt.figure(figsize=fig_size)
            plt.scatter(block_df['TimeEvent'], block_df['Pupil'], s=dot_size)
            plt.xlabel("TimeEvent")
            plt.ylabel("Pupil")
            plt.title(f"Pupil vs TimeEvent - Block {i+1}")
            plt.grid(True, alpha=0.4)
            plt.show()
