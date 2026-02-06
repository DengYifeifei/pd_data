#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import beta as beta_dist
from functions import DataAnalyze_new, get_information_gain, kl_beta

# ------------------------
# Config
# ------------------------
version = 'v1'
eye_directory = f'data/processed/{version}/eyetracking'
main_directory = f'data/processed/{version}'

# available_folders = [
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
available_folders = [
    '25-10-17-1206_setting0',
    '25-10-24-1302_setting0',
    '25-10-30-1254_setting0',
    '25-10-22-1358_setting0',
    '25-09-30-1530_setting0',
    '25-09-30-1338_setting0',
    '25-10-01-1417_setting0',
    '25-11-05-1457_setting0',
    '25-09-29-1238_setting0',
    '25-11-03-1151_setting0',
    '25-11-04-1404_setting0',
    # '25-11-07-1456_setting0',
    '25-11-05-1407_setting0',
    '25-09-30-1427_setting0',
    '25-10-08-1408_setting0',
    '25-10-20-1204_setting0',
    # '25-10-07-1513_setting0',
    '25-10-21-1446_setting0',
    # '25-11-05-1407_setting0',
    '25-10-23-1258_setting0',
    '25-09-30-1142_setting0',
    '25-10-24-1354_setting0',
    '25-10-20-1257_setting0',
    '25-11-03-1301_setting0',
    # '25-11-07-1401_setting0',
    # '25-11-06-1302_setting0'
]

# Model predictor sets (each has 5 predictors)
model_preds = {
    "1b": ["stimulus_consistency", "cue_surprise", "learning_surprise_cued", "cuedStim_surprise_cued", "noncuedStim_surprise_cued"],
    "2b": ["stimulus_consistency", "cue_surprise", "learning_surprise_bi", "cuedStim_surprise_bi", "noncuedStim_surprise_bi"]
}

# Power loop settings
sample_sizes = list(range(27, 40))   # 5..20 inclusive
n_repeats = 100                     # repeats per sample size
n_models = len(model_preds)         # 2
n_predictors = 5                    # each model has 5 predictors

# Multiprocessing
n_workers = os.cpu_count() or 1

# Output file
output_filename = "pupil_power_results.npz"

# ------------------------
# Helper functions
# ------------------------
def cum_counts_cued(block_df):
    block_df = block_df.copy()
    block_df['cumA_beforetrial_cued'] = block_df['is_cued_A'].cumsum().shift(1, fill_value=0) + 1
    block_df['cumB_beforetrial_cued'] = block_df['is_cued_B'].cumsum().shift(1, fill_value=0) + 1
    block_df['cumA_aftertrial_cued'] = block_df['is_cued_A'].cumsum() + 1
    block_df['cumB_aftertrial_cued'] = block_df['is_cued_B'].cumsum() + 1
    return block_df

def cum_counts_bilateral(block_df):
    block_df = block_df.copy()
    block_df['cumA_beforetrial_bi'] = (block_df['is_left_A'] + block_df['is_right_A']).cumsum().shift(1, fill_value=0) + 1
    block_df['cumB_beforetrial_bi'] = (block_df['is_left_B'] + block_df['is_right_B']).cumsum().shift(1, fill_value=0) + 1
    block_df['cumA_aftertrial_bi'] = (block_df['is_left_A'] + block_df['is_right_A']).cumsum() + 1
    block_df['cumB_aftertrial_bi'] = (block_df['is_left_B'] + block_df['is_right_B']).cumsum() + 1
    return block_df

def process_folder(folder, subject_id):
    """
    Load baseline file and compute required derived columns for pupil modeling.
    Returns a DataFrame or None if file missing or error.
    """
    baseline_file = os.path.join(eye_directory, folder, 'baseline_with_pupilSound.csv')
    if not os.path.exists(baseline_file):
        print(f"⚠️ Missing file for {folder}, skipping")
        return None

    try:
        pupil = pd.read_csv(baseline_file)
    except Exception as e:
        print(f"⚠️ Failed reading CSV for {folder}: {e}")
        return None

    try:
        participant = DataAnalyze_new(folder, main_directory)
    except Exception as e:
        print(f"⚠️ Failed creating DataAnalyze_new for {folder}: {e}")
        return None

    # Basic metadata + performance filter will be applied later
    pupil['subject'] = subject_id
    # ensure participant.log_data indices align with pupil rows
    pupil['block_index'] = participant.log_data['block_index']
    pupil['trial_index_within_block'] = participant.log_data['trial_index_within_block']
    pupil['stimulus_consistency'] = participant.log_data['stimulus_consistency']
    pupil['block_entropy'] = participant.log_data['block_entropy']
    pupil['performance'] = participant.log_data['performance']

    # stimulus tokens
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
    }, index=participant.log_data.index)

    # ---- cued counts / learning surprise (cued) ----
    cuedstim = pd.DataFrame({'block_index': participant.log_data['block_index'],
                             'cued_stim': cued_stim}, index=participant.log_data.index)
    cuedstim['is_cued_A'] = (cuedstim['cued_stim']=='A').astype(int)
    cuedstim['is_cued_B'] = (cuedstim['cued_stim']=='B').astype(int)
    cuedstim = cuedstim.groupby('block_index', group_keys=False).apply(cum_counts_cued)
    cuedstim['priorAprob_cued'] = cuedstim['cumA_beforetrial_cued'] / (cuedstim['cumA_beforetrial_cued'] + cuedstim['cumB_beforetrial_cued'])
    cuedstim['priorBeta_cued'] = cuedstim.apply(lambda row: beta_dist(row['cumA_beforetrial_cued'], row['cumB_beforetrial_cued']), axis=1)
    cuedstim['postBeta_cued'] = cuedstim.apply(lambda row: beta_dist(row['cumA_aftertrial_cued'], row['cumB_aftertrial_cued']), axis=1)
    pupil['learning_surprise_cued'] = cuedstim.apply(lambda row: kl_beta(row['priorBeta_cued'], row['postBeta_cued']), axis=1)
    pupil['cuedStim_surprise_cued'] = np.where(stiminfo['cued_stim']=='A', -np.log(cuedstim['priorAprob_cued']), -np.log(1-cuedstim['priorAprob_cued']))
    pupil['noncuedStim_surprise_cued'] = np.where(stiminfo['noncued_stim']=='A', -np.log(cuedstim['priorAprob_cued']), -np.log(1-cuedstim['priorAprob_cued']))

    # ---- bilateral counts / learning surprise (bi) ----
    bilateralstim = pd.DataFrame({'block_index': participant.log_data['block_index'],
                                  'left_stim': left_stim, 'right_stim': right_stim}, index=participant.log_data.index)
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

    # ---- cue surprise ----
    participant.log_data = get_information_gain(participant.log_data)
    cue = participant.log_data['cue_direction']
    subjective_leftcue_prob = participant.log_data['cue_learning_posterior'].apply(lambda b: b.mean())
    cue_prob = np.where(cue=='left', subjective_leftcue_prob, 1 - subjective_leftcue_prob)
    pupil['cue_surprise'] = -np.log(cue_prob)

    return pupil

def fit_pupil_model(all_pupil, predictors):
    """
    Fit mixed model and return the fitted model object.
    """
    df = all_pupil.copy()
    formula_cols = ["max_pupilRatio"] + predictors
    for col in formula_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'stimulus_consistency' in predictors:
        df['stimulus_consistency'] = df['stimulus_consistency'].astype(int)
    clean_data = df.dropna(subset=formula_cols)
    if clean_data.empty:
        raise ValueError("No data after dropping NaNs for formula columns")
    formula = "max_pupilRatio ~ " + " + ".join(predictors)
    model = smf.mixedlm(formula, clean_data, groups=clean_data["subject"]).fit(reml=True)
    return model

# Worker function must be top-level so it can be pickled by ProcessPoolExecutor
def run_single_repeat(sample_size, available_folders, seed):
    """
    Sample 'sample_size' folders (with replacement), build combined DataFrame,
    fit both models (1b and 2b), and return p-values and coefficients arrays.
    Returns:
        (pvals_array, coeffs_array) each shape (n_models, n_predictors)
    """
    random.seed(seed)
    np.random.seed(seed + 12345)

    # pick folders with replacement (like original script)
    included_folders = random.choices(available_folders, k=sample_size)

    # build data for sampled folders
    pupil_data_list = []
    subject_counter = 0
    for folder in included_folders:
        # skip dotfiles
        if folder.startswith('.'):
            continue
        subject_counter += 1
        df = process_folder(folder, subject_counter)
        if df is None:
            continue
        pupil_data_list.append(df)

    if not pupil_data_list:
        # no data, return nan arrays
        return (np.full((n_models, n_predictors), np.nan), np.full((n_models, n_predictors), np.nan))

    all_pupil = pd.concat(pupil_data_list, ignore_index=True)
    all_pupil['max_pupilRatio'] = all_pupil['max_pupilSound'] / all_pupil['baseline']
    # only correct trials (following your previous filtering)
    all_pupil = all_pupil[all_pupil['performance'] == "Correct"].copy()

    # arrays to hold pvals and coeffs for the two models
    pvals_out = np.full((n_models, n_predictors), np.nan)
    coeffs_out = np.full((n_models, n_predictors), np.nan)

    # iterate models in a stable order
    for m_idx, (name, preds) in enumerate(sorted(model_preds.items())):
        try:
            model = fit_pupil_model(all_pupil, preds)
            # extract coefficients and p-values for each predictor (in same order)
            coeffs_out[m_idx, :] = [model.params.get(pred, np.nan) for pred in preds]
            pvals_out[m_idx, :] = [model.pvalues.get(pred, np.nan) for pred in preds]
        except Exception as e:
            # return NaNs for failed fits but print a short warning
            print(f"⚠️ Fit failed for sample_size={sample_size}, model={name}, seed={seed}: {e}")
            coeffs_out[m_idx, :] = np.full(n_predictors, np.nan)
            pvals_out[m_idx, :] = np.full(n_predictors, np.nan)

    return (pvals_out, coeffs_out)

# ------------------------
# Main loop: parallel over repeats for each sample size
# ------------------------
def main():
    # get available folders from eye directory
    # available_folders = [f for f in os.listdir(eye_directory) if not f.startswith('.')]
    if len(available_folders) == 0:
        raise RuntimeError(f"No subject folders found in {eye_directory}")

    num_sizes = len(sample_sizes)

    # Preallocate arrays:
    # shapes: (num_sample_sizes, n_repeats, n_models, n_predictors)
    pvals_all = np.full((num_sizes, n_repeats, n_models, n_predictors), np.nan)
    coeffs_all = np.full((num_sizes, n_repeats, n_models, n_predictors), np.nan)

    print(f"Starting parallel runs: sample sizes {sample_sizes}, repeats={n_repeats}, workers={n_workers}")

    # Outer loop over sample sizes
    for si, ss in enumerate(sample_sizes):
        print(f"\n=== Sample size {ss} ({si+1}/{num_sizes}) ===")
        # Use a fresh executor for this sample size
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for t in range(n_repeats):
                seed = si * 100000 + t  # unique-ish seed per job
                futures.append(executor.submit(run_single_repeat, ss, available_folders, seed))

            # collect results as completed (but we will place them by index t)
            # to maintain ordering we iterate over enumerate(futures) rather than as_completed
            # but to also surface failures we can gather via as_completed mapping
            for idx, future in enumerate(futures):
                try:
                    pvals_out, coeffs_out = future.result()
                    pvals_all[si, idx, :, :] = pvals_out
                    coeffs_all[si, idx, :, :] = coeffs_out
                except Exception as e:
                    print(f"⚠️ Future index {idx} failed for sample_size={ss}: {e}")
                    # leave NaNs in that slot

        # quick progress save after each sample size (so partial results preserved)
        try:
            np.savez_compressed("pupil_power_progress.npz",
                                pvals=pvals_all,
                                coeffs=coeffs_all,
                                sample_sizes=np.array(sample_sizes))
            print(f"✅ Progress saved after sample size {ss} to pupil_power_progress.npz")
        except Exception as e:
            print(f"⚠️ Failed saving progress: {e}")

    # Final save
    np.savez_compressed(output_filename,
                        pvals=pvals_all,
                        coeffs=coeffs_all,
                        sample_sizes=np.array(sample_sizes))
    print(f"\n🎉 All done. Results saved to {output_filename}")

if __name__ == "__main__":
    main()
