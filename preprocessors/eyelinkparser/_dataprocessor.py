import re
import numpy as np
import pandas as pd
import math
import json 

class DataProcessor:
    def __init__(self, trial_dir, eye_dir,output_dir=None):
        self.trial_data = self.read_trial_data(trial_dir)
        self.eye_data = self.read_eye_data(eye_dir)
        self.output_dir = output_dir

    def read_trial_data(self, filepath):
        """ Read trial data from a JSON file. """
        with open(filepath, 'r') as file:
            return pd.read_json(file)
    
    def read_eye_data(self, filepath):
        """ Read eye-tracking data from a CSV file. """
        return pd.read_csv(filepath)
    
    
    def process_trial(self, trial_data, trial_index):
        """ Process a single trial's eye-tracking data. """
        processed_trial_data = []
        for visit in range(4):  # assuming maximum 3 visits per trial
            visit_eye_data = self.eye_data[(self.eye_data['trial_index'] == trial_index) & (self.eye_data['visit'] == visit)]
            fixation_data = self.process_fixation(visit_eye_data)
            saccade_data = self.process_saccade(visit_eye_data)
            gaze_data = self.count_gaze(visit_eye_data)

            processed_trial = {
                **trial_data.to_dict(orient='records')[0],
                'fixation': fixation_data,
                'saccade': saccade_data,
                'gaze': gaze_data,
                'visit': visit,
            }
            processed_trial_data.append(processed_trial)
        return processed_trial_data

    
    def match(self):
        """ Match the trial data with the eye-tracking data. """
        processed_data = []
        for trial_index in self.trial_data['trial_index'].unique():
            trial_data = self.trial_data[self.trial_data['trial_index'] == trial_index]
            trial_data_filtered = trial_data[['graph', 'rewards', 'start', 'choice', 'layer1', 'layer2', 'trial_index', 'difficulty', 'difficulty_1', 'difficulty_2', 'connect_nodes', 'non_connect_nodes','type', 'wid','accuracy',
                'accuracy_1','df', 'RT_first_visit', 'RT_second_visit', 'RT', 'max_reward', 'loss']]


            processed_data.extend(self.process_trial(trial_data_filtered, trial_index))

        return processed_data

    def process_fixation(self, eye_data):
        """ Process fixation data. """
        fixations = eye_data[eye_data['Type'] == 'Fixation']
        return [{'node': row['Node'], 'duration': row['Duration']} for index, row in fixations.iterrows()]

    def count_gaze(self, eye_data):
        """ Count the number of gaze events for each node. """
        gaze_data = eye_data[eye_data['Type'] == 'Gaze']
        node_counts = gaze_data.groupby('Node').size()
        
        # Initialize a dictionary to hold counts for all nodes, defaulting to 0
        all_node_counts = {node: 0 for node in range(-1, 10)}  # Adjust range as necessary

        # Update this dictionary with the actual counts from the data
        for node, count in node_counts.items():
            all_node_counts[node] = count
        
        return all_node_counts

    def process_saccade(self, eye_data):
        """ Process saccade data. """
        saccades = eye_data[eye_data['Type'] == 'Saccade']
        return [{'start_node': row['Start_Node'], 'end_node': row['End_Node'], 'duration': row['Duration']} for index, row in saccades.iterrows()]

