import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import beta as beta_dist
from functions import DataAnalyze_new, get_information_gain, kl_beta
import pickle
from multiprocessing import Pool, cpu_count


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

def cum_counts_keypress(block_df):
    block_df['cumf_beforetrial_pressed'] = block_df['pressed_f'].cumsum().shift(1, fill_value=0) + 1
    block_df['cumj_beforetrial_pressed'] = block_df['pressed_j'].cumsum().shift(1, fill_value=0) + 1
    block_df['cumf_aftertrial_pressed'] = block_df['pressed_f'].cumsum() + 1
    block_df['cumj_aftertrial_pressed'] = block_df['pressed_j'].cumsum() + 1
    block_df['cumf_beforetrial_should'] = block_df['should_press_f'].cumsum().shift(1, fill_value=0) + 1
    block_df['cumj_beforetrial_should'] = block_df['should_press_j'].cumsum().shift(1, fill_value=0) + 1
    block_df['cumf_aftertrial_should'] = block_df['should_press_f'].cumsum() + 1
    block_df['cumj_aftertrial_should'] = block_df['should_press_j'].cumsum() + 1
    return block_df

def pad_or_truncate(list_of_arrays, target_len=70, pad_value=np.nan):
    n = len(list_of_arrays)
    out = np.full((n, target_len), pad_value)

    for i, arr in enumerate(list_of_arrays):
        arr = np.asarray(arr)
        L = min(len(arr), target_len)
        out[i, :L] = arr[:L]

    return out

def process_folder(folder_subject_tuple):
    folder, subject_id = folder_subject_tuple
    
    baseline_file = os.path.join(eye_directory, folder, 'baseline_with_pupilSound.csv')
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(f"Missing baseline file for {folder}")
    
    pupil = pd.read_csv(baseline_file)
    participant = DataAnalyze_new(folder, main_directory)
    print(f"📂 Loading data from {folder} ({len(pupil)} trials)")

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

    # ---- Cued counts ----
    cuedstim = pd.DataFrame({'block_index': participant.log_data['block_index'], 'cued_stim': cued_stim})
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
    bilateralstim = pd.DataFrame({'block_index': participant.log_data['block_index'], 'left_stim': left_stim, 'right_stim': right_stim})
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

    # ---- Keypress surprise ---- 
    keypress_df = pd.DataFrame({'block_index': participant.log_data['block_index'], 'correct_response': participant.log_data['correct_response'], 'participant_response': participant.log_data['response']})
    keypress_df['pressed_f'] = (keypress_df['participant_response']=='f').astype(int)
    keypress_df['pressed_j'] = (keypress_df['participant_response']=='j').astype(int)
    keypress_df['should_press_f'] = (keypress_df['correct_response']=='f').astype(int)
    keypress_df['should_press_j'] = (keypress_df['correct_response']=='j').astype(int)

    keypress_df = keypress_df.groupby('block_index', group_keys=False).apply(cum_counts_keypress) 

    keypress_df['prior_f_prob_pressed'] = keypress_df['cumf_beforetrial_pressed'] / (keypress_df['cumf_beforetrial_pressed'] + keypress_df['cumj_beforetrial_pressed'])
    keypress_df['priorBeta_pressed'] = keypress_df.apply(lambda row: beta_dist(row['cumf_beforetrial_pressed'], row['cumj_beforetrial_pressed']), axis=1)
    keypress_df['postBeta_pressed'] = keypress_df.apply(lambda row: beta_dist(row['cumf_aftertrial_pressed'], row['cumj_aftertrial_pressed']), axis=1)

    pupil['keypress_learning_surprise_did'] = keypress_df.apply(lambda row: kl_beta(row['priorBeta_pressed'], row['postBeta_pressed']), axis=1) 
    pupil['keypress_surprise_did'] = np.where(keypress_df['participant_response']=='f', -np.log(keypress_df['prior_f_prob_pressed']), -np.log(1-keypress_df['prior_f_prob_pressed'])) 

    keypress_df['prior_f_prob_should'] = keypress_df['cumf_beforetrial_should'] / (keypress_df['cumf_beforetrial_should'] + keypress_df['cumj_beforetrial_should'])
    keypress_df['priorBeta_should'] = keypress_df.apply(lambda row: beta_dist(row['cumf_beforetrial_should'], row['cumj_beforetrial_should']), axis=1)
    keypress_df['postBeta_should'] = keypress_df.apply(lambda row: beta_dist(row['cumf_aftertrial_should'], row['cumj_aftertrial_should']), axis=1)

    pupil['keypress_learning_surprise_should'] = keypress_df.apply(lambda row: kl_beta(row['priorBeta_should'], row['postBeta_should']), axis=1) 
    pupil['keypress_surprise_should'] = np.where(keypress_df['correct_response']=='f', -np.log(keypress_df['prior_f_prob_should']), -np.log(1-keypress_df['prior_f_prob_should'])) 

    # ---- Load pupil_timecourse_binned ----
    save_path = os.path.join(eye_directory, folder, "pupil_timecourse_binned.npy")
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"{save_path} not found")
    pupil_timecourse_binned = np.load(save_path, allow_pickle=True)
    if pupil_timecourse_binned is None:
        raise ValueError(f"{folder}: loaded pupil_timecourse_binned is None")
    pupil_timecourse_binned = pad_or_truncate(pupil_timecourse_binned)

    return pupil, pupil_timecourse_binned



# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Prepare folders with subject IDs
    folders_to_process = [(folder, i+1) for i, folder in enumerate(yes_folders)]

    # Multiprocessing
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_folder, folders_to_process)

    # Unpack results
    pupil_data_list, pupil_timecourse_list = zip(*results)
    pupil_data_list = list(pupil_data_list)
    pupil_timecourse_list = list(pupil_timecourse_list)

    all_pupil = pd.concat(pupil_data_list, ignore_index=True)

    n_bins = pupil_timecourse_list[0].shape[1]

    # -----------------------------
    # Define formulas
    # -----------------------------
    formulas_and_names = [
        {
            "name": "regression_results_cuedsidelearning.pkl",
            "predictors": [
                "stimulus_consistency",
                "cue_learning_surprise",
                "cue_surprise",
                "learning_surprise_cued",
                "cuedStim_surprise_cued",
                "noncuedStim_surprise_cued",
                "keypress_learning_surprise_did",
                "keypress_surprise_did"
            ],
            "formula": (
                "pupil_t ~ stimulus_consistency * cue_surprise + "
                "cue_learning_surprise + learning_surprise_cued + "
                "cuedStim_surprise_cued + noncuedStim_surprise_cued + "
                "keypress_learning_surprise_did + keypress_surprise_did"
            )
        },
        {
            "name": "regression_results_2sidelearning_did.pkl",
            "predictors": [
                "stimulus_consistency",
                "cue_learning_surprise",
                "cue_surprise",
                "learning_surprise_bi",
                "cuedStim_surprise_bi",
                "noncuedStim_surprise_bi",
                "keypress_learning_surprise_did",
                "keypress_surprise_did"
            ],
            "formula": (
                "pupil_t ~ stimulus_consistency * cue_surprise + "
                "cue_learning_surprise + learning_surprise_bi + "
                "cuedStim_surprise_bi + noncuedStim_surprise_bi + "
                "keypress_learning_surprise_did + keypress_surprise_did"
            )
        },
        {
            "name": "regression_results_2sidelearning_should.pkl",
            "predictors": [
                "stimulus_consistency",
                "cue_learning_surprise",
                "cue_surprise",
                "learning_surprise_bi",
                "cuedStim_surprise_bi",
                "noncuedStim_surprise_bi",
                "keypress_learning_surprise_should",
                "keypress_surprise_should"
            ],
            "formula": (
                "pupil_t ~ stimulus_consistency * cue_surprise + "
                "cue_learning_surprise + learning_surprise_bi + "
                "cuedStim_surprise_bi + noncuedStim_surprise_bi + "
                "keypress_learning_surprise_should + keypress_surprise_should"
            )
        }
    ]

    # -----------------------------
    # Run regressions
    # -----------------------------
    for item in formulas_and_names:
        predictors = item["predictors"]
        formula_template = item["formula"]
        save_name = item["name"]

        df_base = all_pupil[["subject", "performance"] + predictors].copy()

        coef_matrix = pd.DataFrame(index=range(n_bins))
        pval_matrix = pd.DataFrame(index=range(n_bins))
        mean_group_size = pd.DataFrame(index=range(n_bins), columns=df_base["subject"].unique())

        for t in range(n_bins):
            # DV: baseline-normalized
            y_tc_raw_list = [pupil_binned[:, t] for pupil_binned in pupil_timecourse_list]
            y_tc_raw = np.concatenate(y_tc_raw_list)
            y_tc = y_tc_raw / all_pupil['baseline']

            df = df_base.copy()
            df["pupil_t"] = y_tc
            df = df[df["performance"] == "Correct"].reset_index(drop=True)
            df = df.dropna(subset=["pupil_t"] + predictors).reset_index(drop=True)

            if len(df) == 0:
                print(f"Warning: no data left for time bin {t} ({save_name}), skipping")
                continue

            # Center predictors
            for col in predictors:
                df[col] = df[col] - df[col].mean()

            # Fit model
            model = smf.mixedlm(formula=formula_template, data=df, groups=df["subject"])
            try:
                result = model.fit(method="lbfgs", disp=False)
            except:
                print(f"Warning: model did not converge for time bin {t} ({save_name})")
                continue

            # Initialize columns after first successful fit
            if coef_matrix.shape[1] == 0:
                coef_matrix = pd.DataFrame(index=range(n_bins), columns=result.params.index)
                pval_matrix = pd.DataFrame(index=range(n_bins), columns=result.pvalues.index)

            # Store results
            for param in result.params.index:
                coef_matrix.loc[t, param] = result.params[param]
                pval_matrix.loc[t, param] = result.pvalues[param]

            # Store mean group size
            group_counts = df.groupby("subject").size()
            for subj in group_counts.index:
                mean_group_size.loc[t, subj] = group_counts[subj]

        # Save results
        results_dict = {
            "coefficients": coef_matrix,
            "p_values": pval_matrix,
            "mean_group_size": mean_group_size
        }

        with open(save_name, "wb") as f:
            pickle.dump(results_dict, f)
        print(f"Saved results to {save_name}")