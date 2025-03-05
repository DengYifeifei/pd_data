import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os
import random

from scipy.stats import linregress


def adjust_trial_index(eye_data):
    """
    Adjust the trial index so that it increases sequentially across blocks.
    Assumes that each block contains 30 trials and the trial index resets at each block.
    """
    previous_i = 0  # Track previous trial index
    block_count = 0  # Track block count

    for i, trial in enumerate(eye_data['trial_index']):
        if trial < previous_i:  # If trial index resets, increment block count
            block_count += 1
        
        # Update trial index to be continuous
        eye_data.at[i, 'trial_index'] = trial + (block_count * 30)
        
        previous_i = trial  # Update previous trial index for next iteration

    return eye_data


def fetch_log(path):
    complete_path = os.path.join('data', 'processed', 'v1', 'logs', path + '.json')
    
    try:
        with open(complete_path, 'r', encoding='utf-8') as f:  # Explicit encoding
            file = json.load(f)
            return pd.DataFrame(file)
    except FileNotFoundError:
        print(f"Error: The file '{complete_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{complete_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def calculate_pupil_differences(df):
    """
    Calculate differences and derivative for pupil data.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Pupil' and 'TimeEvent' columns.

    Returns:
    pd.DataFrame: Updated DataFrame with 'Pupil_diff', 'TimeEvent_diff', and 'Pupil_derivative' columns.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    
    # Calculate pupil size difference
    df['Pupil_diff'] = df['Pupil'].shift(-1) - df['Pupil']
    
    # Calculate time difference
    df['TimeEvent_diff'] = df['TimeEvent'].shift(-1) - df['TimeEvent']
    
    # Calculate pupil size derivative
    df['Pupil_derivative'] = df['Pupil_diff'] / df['TimeEvent_diff']
    
    # Drop NaN values (last row will have NaN due to shifting)
    df.dropna(subset=['Pupil_derivative'], inplace=True)
    
    return df

def record_blink(eye_data):
    blink_data = eye_data[eye_data['Type'] == 'Blink'][['Start', 'End']].copy()
    return blink_data

def find_pupil_bounds_around_blink(eye_data, blink_data, inner_bound=0.2, outer_bound=0.22):
    before_pupil_sizes = []
    after_pupil_sizes = []

    for _, blink in blink_data.iterrows():
        start_time = blink['Start']
        end_time = blink['End']

        before_data = eye_data[(eye_data['TimeEvent'] >= start_time - outer_bound) &
                                  (eye_data['TimeEvent'] < start_time - inner_bound)]
        avg_before = before_data['Pupil'].mean()

        after_data = eye_data[(eye_data['TimeEvent'] > end_time + inner_bound) &
                                 (eye_data['TimeEvent'] <= end_time + outer_bound)]
        avg_after = after_data['Pupil'].mean()

        before_pupil_sizes.append(avg_before)
        after_pupil_sizes.append(avg_after)
    return before_pupil_sizes, after_pupil_sizes

def select_valid_pupil_size(eye_data, blink_data, check_bound=0.2):

    pupil_data = eye_data[eye_data['Pupil'].notna()]
    #print("before blink process:", len(pupil_data))

    selected_data = []

    for _, row in pupil_data.iterrows():
        valid = True
        for _, blink in blink_data.iterrows():
            if blink['Start'] - check_bound <= row['TimeEvent'] <= blink['End'] + check_bound:
                #print("checking validity")
                avg_before = blink['Avg_Pupil_Before']
                avg_after = blink['Avg_Pupil_After']
                if not (avg_before <= row['Pupil'] <= avg_after):
                    valid = False
                    break
        if valid:
            selected_data.append(row)

    selected_data = pd.DataFrame(selected_data)
    return selected_data

def plot_pupil_size(pupil_data,size=3, figsize = (10,6)):
    if 'TimeEvent' in pupil_data.columns and 'Pupil' in pupil_data.columns:
        if len(pupil_data['trial_index'].unique()) == 1:
            trial_index = pupil_data['trial_index'].iloc[0]
        else:
            trial_index = "Multiple trials"  
        plt.figure(figsize=figsize)
        plt.scatter(pupil_data['TimeEvent'], pupil_data['Pupil'], color='blue', alpha=0.6,s=size, label='Pupil Size')
        plt.title(f'Pupil Size Over TimeEvent - Trial {trial_index}')        
        plt.xlabel('TimeEvent')
        plt.ylabel('Pupil Size')
        plt.legend()
        plt.grid(True)
        plt.show()


def KLdivergence(prior, posterior, epsilon=1e-2):
    prior = np.array(prior)
    posterior = np.array(posterior)
    
    prior = np.where(prior <= 0, epsilon, prior)
    posterior = np.where(posterior <= 0, epsilon, posterior)
    
    divergence = 0
    for i in range(len(prior)):
        divergence += posterior[i] * np.log(posterior[i] / prior[i])
    
    return divergence


def sensory_posterior(stimulus):
    mapping = {
        'A|A': [1, 0, 0, 0],
        'A|B': [0, 1, 0, 0],
        'B|A': [0, 0, 1, 0],
        'B|B': [0, 0, 0, 1]
    }
    if stimulus not in mapping:
        print('stimulus not in mapping')
    return mapping.get(stimulus, [0, 0, 0, 0])  # Default case if unknown stimulus

def sensory_prior(stimA_prob):
    return [stimA_prob**2, stimA_prob*(1 - stimA_prob), 
            stimA_prob*(1 - stimA_prob), (1 - stimA_prob)**2]

def action_prior(A_probability):
    return [A_probability, 1 - A_probability]


def action_posterior(stimulus, cue_direction):
    mapping = {
        'A|A': [1, 0], 
        'A|B': [1, 0] if cue_direction == 'left' else [0, 1],  
        'B|A': [0, 1] if cue_direction == 'left' else [1, 0],  
        'B|B': [0, 1] 
    }
    
    if stimulus not in mapping:
        print(f"⚠️ Warning: Unknown stimulus '{stimulus}' encountered!")
    
    return mapping.get(stimulus, [0, 0])  

def action_cue_prior(A_prob, leftCue_prob):
    return [A_prob*leftCue_prob, A_prob*(1-leftCue_prob), (1 - A_prob)*leftCue_prob, (1 - A_prob)*(1-leftCue_prob)]
'''or correct_response? '''
def action_cue_posterior(response, cue_direction):
    if response == 'f':
        A_prob = 1
    else:
        A_prob = 0
    
    if cue_direction == 'left': 
        leftCue_prob = 1
    else:
        leftCue_prob = 0 

    return action_cue_prior(A_prob,leftCue_prob)


def sensory_cue_prior(A_prob, leftCue_prob):
    store_sensory_prior = sensory_prior(A_prob)
    prior = [leftCue_prob * val for val in store_sensory_prior] + \
            [(1 - leftCue_prob) * val for val in store_sensory_prior]
    return prior

def sensory_cue_posterior(stimulus, cue_direction):
    store_sensory = sensory_posterior(stimulus)
    
    if cue_direction == 'left': 
        leftCue_prob = 1
    else:
        leftCue_prob = 0 
    
    posterior = [leftCue_prob * val for val in store_sensory] + \
            [(1 - leftCue_prob) * val for val in store_sensory]

    return posterior  



class DataAnalyze_new:
    def __init__(self, file_name) -> None:
        if os.path.exists(f'data/processed/v1/eyetracking/{file_name}/all_adin.csv'):  # Fixed typo: missing parentheses
            self.eye_raw = pd.read_csv(f'data/processed/v1/eyetracking/{file_name}/all_adin.csv')
        else:
            _eye_data = pd.read_csv(f'data/processed/v1/eyetracking/{file_name}/all.csv')
            self.eye_raw = adjust_trial_index(_eye_data) 
            self.eye_raw.to_csv(f"data/processed/v1/eyetracking/{file_name}/all_adin.csv", index=False)
        print("eye data obtained")
        self.log_data = fetch_log(file_name)
        print("log data obtained")

        self.num_trials = len(self.log_data['trial_index'].unique())
        self.pupil_statistics_raw = np.zeros((self.num_trials, 3, 2))  # 3 events, 2 statistics (max, mean)

    def show_performance(self):
        '''incorrect trial'''
        self.incorrect_count = (self.log_data["performance"] == "Incorrect").sum()
        self.incorrect_trial_list = self.log_data.loc[self.log_data["performance"] == "Incorrect", "trial_index"]
        print("incorrect count:", self.incorrect_count, "incorrect trial list:", self.incorrect_trial_list)

        '''rt plot'''
        bins = np.arange(0, self.log_data["rt"].max() + 0.5, 0.5)  # Binning in 0.5s intervals (adjust if needed)

        # Create a histogram
        plt.hist(self.log_data["rt"], bins=bins, edgecolor='black', alpha=0.7)

        # Labels and title
        plt.xlabel("rt (seconds)")
        plt.ylabel("Frequency")
        plt.title("rt Distribution")
        plt.grid(axis='y', linestyle="--", alpha=0.7)

        # Show the plot
        plt.show()
    
    def filter_eye_data_diff_blink(self, eye_data, trial_index_list, rep = 3, dot_size = 4, diff_lower_bound=-2, diff_upper_bound=2, alpha_inner=0.5, alpha_outer = 0.5, inner_bound=0.2, outer_bound=0.22, plot = False):

        eye_data = eye_data[eye_data['trial_index'].isin(trial_index_list)].copy()

        blink_data = record_blink(eye_data)
        #print(blink_data)


        for rep in range(rep):
    
            #find reasonable pupil size bound
            Avg_Pupil_Before, Avg_Pupil_After = find_pupil_bounds_around_blink(eye_data, blink_data, inner_bound=inner_bound, outer_bound=outer_bound)
            # print("Pupil bounds before blink: ", Avg_Pupil_Before, "Pupil bounds after blink: ", Avg_Pupil_After)
            blink_data['Avg_Pupil_Before'] = Avg_Pupil_Before
            blink_data['Avg_Pupil_After'] = Avg_Pupil_After

            # exclude unreasonable pupil size
            eye_data = select_valid_pupil_size(eye_data, blink_data)
            # print("row numbers after blink process: ", len(eye_data))
            
            #exclude unreasonable differences
            eye_data = calculate_pupil_differences(eye_data)

            eye_data = eye_data[
                (eye_data['Pupil_diff'] >= diff_lower_bound) &
                (eye_data['Pupil_diff'] <= diff_upper_bound) 
            ]

            #print("row number after diff filtering: ", len(eye_data))
            # plot_pupil_size(eye_data)
            
            #print("inner bound:", inner_bound,"outer bound", outer_bound)

            if rep == 0: 
                inner_bound=alpha_inner*inner_bound
                outer_bound=alpha_outer*outer_bound

        
        if plot:
            plot_pupil_size(eye_data, size=dot_size)

        return eye_data

    def get_surprise_level_data(self):
        # Define the necessary calculations for surprise levels
        
        # Compute action prior and posterior surprises
        self.log_data['action_prior'] = self.log_data['left_cue_condition'].apply(action_prior)
        self.log_data['action_posterior'] = self.log_data.apply(
            lambda row: action_posterior(row['stimulus'], row['cue_direction']), axis=1
        )

        # Compute KL divergence for action surprise
        self.log_data['action_surprise'] = self.log_data.apply(
            lambda row: KLdivergence(row['action_prior'], row['action_posterior']), axis=1
        )

        # Compute sensory prior and posterior surprises
        self.log_data['sensory_prior'] = self.log_data['stimA_condition'].apply(sensory_prior)
        self.log_data['sensory_posterior'] = self.log_data['stimulus'].apply(sensory_posterior)

        # Compute KL divergence for sensory surprise
        self.log_data['sensory_surprise'] = self.log_data.apply(
            lambda row: KLdivergence(row['sensory_prior'], row['sensory_posterior']), axis=1
        )

        # Compute action cue prior and posterior surprises
        self.log_data['action_cue_prior'] = self.log_data.apply(
            lambda row: action_cue_prior(row['stimA_condition'], row['left_cue_condition']), axis=1
        )

        self.log_data['action_cue_posterior'] = self.log_data.apply(
            lambda row: action_cue_posterior(row['response'], row['left_cue_condition']), axis=1
        )

        # Compute KL divergence for action cue surprise
        self.log_data['action_cue_surprise'] = self.log_data.apply(
            lambda row: KLdivergence(row['action_cue_prior'], row['action_cue_posterior']), axis=1
        )

        # Compute sensory cue prior and posterior surprises
        self.log_data['sensory_cue_posterior'] = self.log_data.apply(
            lambda row: sensory_cue_posterior(row['stimulus'], row['left_cue_condition']), axis=1
        )
        
        self.log_data['sensory_cue_prior'] = self.log_data.apply(
            lambda row: sensory_cue_prior(row['stimA_condition'], row['left_cue_condition']), axis=1
        )

        # Compute KL divergence for sensory cue surprise
        self.log_data['sensory_cue_surprise'] = self.log_data.apply(
            lambda row: KLdivergence(row['sensory_cue_prior'], row['sensory_cue_posterior']), axis=1
        )

        # Store the surprise levels in trial_surprise
        keys = ['action_surprise', 'sensory_surprise', 'action_cue_surprise', 'sensory_cue_surprise']
        num_surpise_models = len(keys)

        self.surprise_data = np.zeros((self.num_trials, num_surpise_models))
        for index, key in enumerate(keys):
            self.surprise_data[:, index] = self.log_data[key]

        return self.surprise_data
    
    def get_pupil_statistics_raw(self):

        # Loop through each trial
        for trial_index in range(1, self.num_trials + 1):

            if np.all(self.pupil_statistics_raw[trial_index-1, :, :] != 0):
                continue
            else:
                print("processing trial", trial_index)
                # Get the data for the current trial, ensuring that Pupil values are not NaN
                _trial_data = self.eye_raw[self.eye_raw['trial_index'] == trial_index].copy()
                
                # List of events we care about ('cue', 'sound', and 'response')
                events = ['cue', 'sound', 'response']
                for event in events:
                    # Map events to labels and event numbers
                    if event == 'sound':
                        label = 'start decision window'
                        event_num = 1
                    elif event == 'cue':
                        label = 'show cue'
                        event_num = 0
                    elif event == 'response':
                        label = 'response'
                        event_num = 2
                        if trial_index == 29:
                            print(_trial_data[_trial_data['event'] == label].dropna(subset=['TimeEvent']))
                    # Try to get the event time for the current event in the trial
                    try:
                        event_time = _trial_data[_trial_data['event'] == label].dropna(subset=['TimeEvent'])['TimeEvent'].iloc[0]
                    except IndexError:
                        raise ValueError(f"No valid 'TimeEvent' found in trial {trial_index} for event {event}.")
            
                    # Define the time window for extracting pupil data
                    start_bound = event_time + 0.5  # Start of the window (0.5 seconds after the event)
                    end_bound = event_time + 2      # End of the window (2 seconds after the event)

                    trial_data = _trial_data[_trial_data['Pupil'].notna()].copy()

                    # Apply filtering to remove blinks or any unwanted data
                    processed_trial_data = self.filter_eye_data_diff_blink(trial_data, [trial_index], plot=False)

                    # Filter the processed trial data to only include the relevant time window
                    processed_event_data = processed_trial_data[
                        (processed_trial_data['TimeEvent'] >= start_bound) & 
                        (processed_trial_data['TimeEvent'] <= end_bound)
                    ]

                    # Check if any data exists for the time window
                    if processed_event_data.empty:
                        raise ValueError(f"Error: processed_event_data is empty for trial {trial_index} and event {event}. Check filtering conditions.")

                    # Calculate the pupil statistics for the current event
                    pupil_max = processed_event_data['Pupil'].max()  # Max pupil size in the time window
                    pupil_mean = processed_event_data['Pupil'].mean()  # Mean pupil size in the time window

                    # Store the statistics (max, mean) in the correct spot in the array
                    self.pupil_statistics_raw[trial_index - 1, event_num, 0] = pupil_max
                    self.pupil_statistics_raw[trial_index - 1, event_num, 1] = pupil_mean

        return self.pupil_statistics_raw
    
    def get_pupil_max(self):


    def generate_masks(data, num_filter = 4):
    num_filter = 4
    index_range = np.zeros((num_filter+1, int(len(data) / 30)))  # Make sure dimensions match
    # print(index_range)
    index_range[0, :] = np.arange(0, len(data), 30)  # Adjust the range

    # Adjust the rest of the rows
    for i in range(1, num_filter+1):
        index_range[i, :] = index_range[i - 1, :] + 5
        
    index_ranges =[]
    # Extract the first and last row from index_range
    for i in range(index_range.shape[0]):
        start_indices = index_range[0, :].astype(int)
        end_indices = index_range[i, :].astype(int)

    # Create list of tuples pairing corresponding elements
        index_ranges.append(list(zip(start_indices, end_indices)))

    # Print result
    # print(index_ranges)

    # Initialize an empty list to store excluded indices for each exclusion method
    all_exclude_indices = []

    # Iterate over each exclusion method in index_ranges
    for i, ranges in enumerate(index_ranges):
        if i == 0:
            exclude_indices = []  # No exclusion for the first case
        else:
            # Flatten the current list of (start, end) tuples
            exclude_indices = [idx for idx in range(len(x)) if any(start <= idx < end for start, end in ranges)]
        
        all_exclude_indices.append(exclude_indices)  # Store excluded indices for this exclusion method

    # Print result
    return all_exclude_indices

