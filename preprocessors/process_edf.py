# import os
# import sys
# import json
# import pandas as pd
# import subprocess

# from config import VERSION
# wid = '25-01-27-1603_setting0'

# if len(sys.argv) > 1:
#     VERSION = sys.argv[1]

# trials = []
# for file in sorted(os.listdir(f"data/exp/{VERSION}/")):
#     if 'test' in file or 'txt' in file:
#         continue
#     # wid = file.replace('.json', '')

    
#     # experimental data
#     # fn = f"data/exp/{VERSION}/{file}"
#     # print(fn)
#     # with open(fn) as f:
#     #     data = json.load(f)
#     # for i, t in enumerate(data["trial_data"]):
#     #     t["wid"] = wid
#     #     t["trial_index"] = i
#     #     trials.append(t)

#     # eyelink data
 
#     edf = f'data/eyelink/{wid}/raw.edf'
#     print("EDF file path:", edf)
#     assert os.path.isfile(edf)
#     dest = f'data/eyelink/{wid}/samples.asc'
#     if os.path.isfile(edf) and not os.path.isfile(dest):
#         cmd = f'edf2asc {edf} {dest}'
#         output = subprocess.getoutput(cmd)
#         if 'Converted successfully' not in output:
#             print(f'Error parsing {edf}', '-'*80, output, '-'*80, sep='\n')


# # os.makedirs(f'data/processed/{VERSION}/', exist_ok=True)
# # with open(f'data/processed/{VERSION}/trials.json', 'w') as f:
# #     json.dump(trials, f)


import os
import sys
import json
import subprocess

from config import VERSION  # Ensure config.py exists
wid = '25-03-07-1109_setting0'

if len(sys.argv) > 1:
    VERSION = sys.argv[1]

trials = []
exp_path = f"data/exp/{VERSION}/"
eyelink_path = f"data/eyelink/{wid}/"

if not os.path.isdir(exp_path):
    print(f"Error: Experiment folder not found: {exp_path}")
    sys.exit(1)

if not os.path.isdir(eyelink_path):
    print(f"Error: Eyelink folder not found: {eyelink_path}")
    sys.exit(1)

# Iterate over all EDF files in the folder
edf_files = [f for f in os.listdir(eyelink_path) if f.endswith('.edf')]

if not edf_files:
    print(f"Warning: No EDF files found in {eyelink_path}")
else:
    for edf_file in edf_files:
        edf_path = os.path.join(eyelink_path, edf_file)
        asc_path = os.path.join(eyelink_path, edf_file.replace('.edf', '.asc'))

        print("Processing EDF file:", edf_path)

        # Convert only if the ASC file does not already exist
        if not os.path.isfile(asc_path):
            cmd = f'edf2asc "{edf_path}" "{asc_path}"'
            output = subprocess.getoutput(cmd)

            if not os.path.isfile(asc_path):  # Check if conversion was successful
                print(f"Error converting {edf_path}", '-'*80, output, '-'*80, sep='\n')

print("Processing complete.")


