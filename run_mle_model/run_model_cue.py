

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import beta as beta_dist
from functions import DataAnalyze_new, get_information_gain, kl_beta


version = 'v1'
eye_directory = f'data/processed/{version}/eyetracking'
main_directory = f'data/processed/{version}'

yes_folders = [
    '25-09-29-1238_setting0',
    '25-09-30-1142_setting0',
    '25-09-30-1338_setting0',
    '25-09-30-1427_setting0',
    '25-09-30-1530_setting0',
    '25-10-01-1417_setting0',
    '25-10-07-1233_setting0',  # yes, maybe no
    '25-10-07-1328_setting0',  # yes?
    '25-10-07-1513_setting0',
    '25-10-09-1303_setting0',
    '25-10-17-1206_setting0',
    '25-10-20-1204_setting0',
    '25-10-20-1257_setting0',
    '25-10-21-1446_setting0',
    '25-10-22-1358_setting0',
    '25-10-23-1258_setting0',  # yes, maybe no
    '25-10-24-1302_setting0',
    '25-10-27-1254_setting0',
    '25-10-30-1254_setting0',
    '25-10-31-1350_setting0',
    '25-11-03-1100_setting0',  # yes?
    '25-11-03-1151_setting0',
    '25-11-03-1301_setting0',  # yes?
    '25-11-04-1214_setting0',
    '25-11-04-1404_setting0',  # yes?
    '25-11-05-1407_setting0',
    '25-11-05-1457_setting0',
    '25-11-06-1302_setting0',  # yes?
    '25-11-07-1401_setting0',
    '25-11-07-1456_setting0',
]

# ------------------------
# Functions
# ------------------------
def cum_counts_cued(block_df):
    block_df['cumA_beforetrial_cued'] = block_df['is_cued_A'].cumsum().shift(1, fill_value=0) + 1
    block_df['cumB_beforetrial_cued'] = block_df['is_cued_B'].cumsum().shift(1, fill_value=0) + 1
    block_df['cumA_aftertrial_cued'] = block_df['is_cued_A'].cumsum() + 1
    block_df['cumB_aftertrial_cued'] = block_df['is_cued_B'].cumsum() + 1
    return block_df

def cum_counts_bilateral(block_df):
    block_df['cumA_beforetrial_bi'] = (block_df['is_left_A'] + block_df['is_right_A']).cumsum().shift(1, fill_value=0) + 1
    block_df['cumB_beforetrial_bi'] = (block_df['is_left_B'] + block_df['is_right_B']).cumsum().shift(1, fill_value=0) + 1
    block_df['cumA_aftertrial_bi'] = (block_df['is_left_A'] + block_df['is_right_A']).cumsum() + 1
    block_df['cumB_aftertrial_bi'] = (block_df['is_left_B'] + block_df['is_right_B']).cumsum() + 1
    return block_df

def process_folder(folder, subject_id):
    """Process one folder and return a DataFrame with pupil data and computed variables."""
    baseline_file = os.path.join(eye_directory, folder, 'baseline_with_pupilSound_response.csv')
    if not os.path.exists(baseline_file):
        print(f"⚠️ Missing file for {folder}, skipping")
        return None
    
    pupil = pd.read_csv(baseline_file)
    participant = DataAnalyze_new(folder, main_directory)
    print(f"📂 Loading data from {folder}")

    # DEBUG: Check initial sizes
    print(f"🔍 DEBUG: Initial sizes")
    print(f"  pupil (baseline file): {len(pupil)} rows")
    print(f"  participant.log_data: {len(participant.log_data)} rows")

    # Add metadata
    pupil['subject'] = subject_id
    pupil['block_index'] = participant.log_data['block_index']
    pupil['trial_index_within_block'] = participant.log_data['trial_index_within_block']
    pupil['stimulus_consistency'] = participant.log_data['stimulus_consistency']
    pupil['block_entropy'] = participant.log_data['block_entropy']
    pupil['performance'] = participant.log_data['performance']


    left_stim = participant.log_data['stimulus'].str[0]
    right_stim = participant.log_data['stimulus'].str[2]
    cued_stim = np.where(participant.log_data['cue_direction']=='left', left_stim, right_stim)
    noncued_stim = np.where(participant.log_data['cue_direction']=='right', left_stim, right_stim)

    stiminfo = pd.DataFrame({
        'block_index': participant.log_data['block_index'],
        'left_stim': left_stim,
        'right_stim': right_stim,
        'cued_stim': pd.Series(cued_stim),
        'noncued_stim': pd.Series(noncued_stim)
    })

    # print(f"🔍 DEBUG: cuedstim before groupby: {len(cuedstim)} rows")


    # ---- Cued counts ----
    cuedstim = pd.DataFrame({'block_index': participant.log_data['block_index'],
                             'cued_stim': cued_stim})
    cuedstim['is_cued_A'] = (cuedstim['cued_stim']=='A').astype(int)
    cuedstim['is_cued_B'] = (cuedstim['cued_stim']=='B').astype(int)
    cuedstim = cuedstim.groupby('block_index', group_keys=False).apply(cum_counts_cued)
    cuedstim['priorAprob_cued'] = cuedstim['cumA_beforetrial_cued'] / (cuedstim['cumA_beforetrial_cued'] + cuedstim['cumB_beforetrial_cued'])
    cuedstim['priorBeta_cued'] = cuedstim.apply(lambda row: beta_dist(row['cumA_beforetrial_cued'], row['cumB_beforetrial_cued']), axis=1)
    cuedstim['postBeta_cued'] = cuedstim.apply(lambda row: beta_dist(row['cumA_aftertrial_cued'], row['cumB_aftertrial_cued']), axis=1)
    pupil['learning_surprise_cued'] = cuedstim.apply(lambda row: kl_beta(row['priorBeta_cued'], row['postBeta_cued']), axis=1)
    pupil['cuedStim_surprise_cued'] = np.where(stiminfo['cued_stim']=='A', -np.log(cuedstim['priorAprob_cued']), -np.log(1-cuedstim['priorAprob_cued']))
    pupil['noncuedStim_surprise_cued'] = np.where(stiminfo['noncued_stim']=='A', -np.log(cuedstim['priorAprob_cued']), -np.log(1-cuedstim['priorAprob_cued']))

    # ---- Bilateral counts ----
    bilateralstim = pd.DataFrame({'block_index': participant.log_data['block_index'],
                                  'left_stim': left_stim, 'right_stim': right_stim})
    bilateralstim['is_left_A'] = (bilateralstim['left_stim']=='A').astype(int)
    bilateralstim['is_left_B'] = (bilateralstim['left_stim']=='B').astype(int)
    bilateralstim['is_right_A'] = (bilateralstim['right_stim']=='A').astype(int)
    bilateralstim['is_right_B'] = (bilateralstim['right_stim']=='B').astype(int)
    bilateralstim = bilateralstim.groupby('block_index', group_keys=False).apply(cum_counts_bilateral)
    bilateralstim['priorAprob_bi'] = bilateralstim['cumA_beforetrial_bi'] / (bilateralstim['cumA_beforetrial_bi'] + bilateralstim['cumB_beforetrial_bi'])
    bilateralstim['priorBeta_bi'] = bilateralstim.apply(lambda row: beta_dist(row['cumA_beforetrial_bi'], row['cumB_beforetrial_bi']), axis=1)
    bilateralstim['postBeta_bi'] = bilateralstim.apply(lambda row: beta_dist(row['cumA_aftertrial_bi'], row['cumB_aftertrial_bi']), axis=1)
    pupil['learning_surprise_bi'] = bilateralstim.apply(lambda row: kl_beta(row['priorBeta_bi'], row['postBeta_bi']), axis=1)
    pupil['cuedStim_surprise_bi'] = np.where(stiminfo['cued_stim']=='A', -np.log(bilateralstim['priorAprob_bi']), -np.log(1-bilateralstim['priorAprob_bi']))
    pupil['noncuedStim_surprise_bi'] = np.where(stiminfo['noncued_stim']=='A', -np.log(bilateralstim['priorAprob_bi']), -np.log(1-bilateralstim['priorAprob_bi']))

    # ---- Cue surprise ----
    participant.log_data = get_information_gain(participant.log_data)
    cue = participant.log_data['cue_direction']
    subjective_leftcue_prob = participant.log_data['cue_learning_prior'].apply(lambda b: b.mean())
    pupil['cue_learning_surprise'] = participant.log_data.apply(lambda row: kl_beta(row['cue_learning_prior'], row['cue_learning_posterior']), axis=1)
    cue_prob = np.where(cue=='left', subjective_leftcue_prob, 1-subjective_leftcue_prob)
    pupil['cue_surprise'] = -np.log(cue_prob)

    return pupil

# ------------------------
# Fit mixed-effects model
# ------------------------
def fit_pupil_model(all_pupil, predictors):
    formula_cols = ["max_pupilcue_Ratio"] + predictors
    for col in formula_cols:
        all_pupil[col] = pd.to_numeric(all_pupil[col], errors='coerce')
    if 'stimulus_consistency' in predictors:
        all_pupil['stimulus_consistency'] = all_pupil['stimulus_consistency'].astype(int)
    clean_data = all_pupil.dropna(subset=formula_cols)
    formula = "max_pupilcue_Ratio ~ " + " + ".join(predictors)
    model = smf.mixedlm(formula, clean_data, groups=clean_data["subject"]).fit(reml=True)
    return model

# ------------------------
# Main execution
# ------------------------
if __name__ == "__main__":
    pupil_data_list = []
    subject_count = 0

    for folder in yes_folders:
        subject_count += 1
        pupil = process_folder(folder, subject_count)
        if pupil is not None:
            pupil_data_list.append(pupil)

    all_pupil = pd.concat(pupil_data_list, ignore_index=True)
    all_pupil['max_pupilcue_Ratio'] = all_pupil['max_pupilCue'] / all_pupil['baseline']
    all_pupil = all_pupil[all_pupil['performance']=="Correct"].copy()
    # all_pupil = all_pupil[all_pupil['trial_index_within_block']<20].copy()

    ''' Keep only correct trials'''
    all_pupil_correct = all_pupil[all_pupil['performance']=="Correct"].copy()

    # Get list of subject IDs that survived
    subjects_present = all_pupil_correct['subject'].unique()

    # Map subject IDs to folders
    subject_folder_map = {i+1: folder for i, folder in enumerate(yes_folders)}

    # Check which folders are missing
    missing_folders = [folder for i, folder in subject_folder_map.items() if i not in subjects_present]

    print("Folders with usable data:", [subject_folder_map[i] for i in subjects_present])
    print("Folders completely missing / dropped:", missing_folders)
    '''end'''

 
    # ---- Define predictor sets ----
    model_preds = {
        # "1a": ["stimulus_consistency", "cue_surprise", "learning_surprise_cued", "cuedStim_surprise_cued"],
        "1": ["cue_surprise", "cue_learning_surprise"],
        # "2a": ["stimulus_consistency", "cue_surprise", "learning_surprise_bi", "cuedStim_surprise_bi"],
        # "2b": ["stimulus_consistency", "cue_surprise", "learning_surprise_bi", "cuedStim_surprise_bi", "noncuedStim_surprise_bi"]
    }

    # ---- Fit and print models ----
    models = {}
    for name, preds in model_preds.items():
        models[name] = fit_pupil_model(all_pupil, preds)
        print(f"\n===== Model {name} =====")
        print(models[name].summary())
