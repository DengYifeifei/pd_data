import numpy as np
import sys
import json
import pandas as pd
import subprocess
from config import VERSION
import os
from eyelinkparser import EyeLinkParser
from eyelinkparser import TrialProcessor as tp

# Use version number provided as an argument if available
if len(sys.argv) > 1:
    VERSION = sys.argv[1]

# Instantiate the TrialProcessor and EyeLinkParser with the version
trial_processor = tp(VERSION)


def save_as_csv(data, filepath):
    # Convert dictionary to DataFrame and save as CSV
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    else:
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)


def main():
    exp_dir = f"data/exp/{VERSION}/"
    eyetrack_dir = f"data/eyelink/"
    processed_trial_dir = f"data/processed/{VERSION}/trial_data/"
    processed_eye_dir = f"data/processed/{VERSION}/eyetracking/"

    # Ensure output directories exist
    os.makedirs(processed_trial_dir, exist_ok=True)
    os.makedirs(processed_eye_dir, exist_ok=True)

    
    # # Process experimental trial data
    for file in sorted(os.listdir(exp_dir)):
        # if 'test' in file or 'txt' in file:
        #     continue
        if not file.endswith('.json'):
            print(f"Skipping non-JSON file: {file}")
            continue
        fn = os.path.join(exp_dir, file)
        print(f"Processing trial data: {fn}")
        processed_data = trial_processor.process_file(fn)
        if processed_data:
            wid = file.replace('.json', '')
            output_path = os.path.join(processed_trial_dir, f'{wid}.json')
            trial_processor.save_data(processed_data, output_path)
            print(f"Trial data saved to {output_path}")

    # Process eye-tracking data
    for participant_dir in sorted(os.listdir(eyetrack_dir)):
        participant_path = os.path.join(eyetrack_dir, participant_dir)
        
        processed_part_dir = os.path.join(processed_eye_dir, participant_dir)
        os.makedirs(processed_part_dir, exist_ok=True)
        all_data = []  # List to store data from all ASC files

        if os.path.isdir(participant_path):
            for file in sorted(os.listdir(participant_path)):
                if file.endswith('.asc'):
                    asc_file = os.path.join(participant_path, file)
                    parser = EyeLinkParser(eye_folder=participant_path, asc_encoding='ISO-8859-1')
                    processed_eye_data = parser.parse_asc_file(asc_file)
                    print(f"Processed data from {file} for {participant_dir}")

                    all_data.append(processed_eye_data)  # Append parsed data

            if all_data:
                # Assuming processed_eye_data is a DataFrame, concatenate all
                all_combined = pd.concat(all_data, ignore_index=True)
                output_eye_file = os.path.join(processed_part_dir, 'all.csv')
                all_combined.to_csv(output_eye_file, index=False)
                print(f"All eye-tracking data saved to {output_eye_file}")
            
    

if __name__ == "__main__":
    main()
