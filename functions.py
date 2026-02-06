import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os
import json
import statsmodels.formula.api as smf
from scipy.special import betaln, digamma, gammaln
from scipy.stats import beta, mannwhitneyu, linregress
import seaborn as sns
from typing import List, Tuple
from scipy.stats import ttest_ind




def convert_nested_ndarray_to_list(array):
    if isinstance(array, np.ndarray):
        return array.tolist()
    elif isinstance(array, list):
        return [a.tolist() if isinstance(a, np.ndarray) else a for a in array]
    return array

'''preprocessing and cleaning'''

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


def add_block_columns(log_data, block_len=30):
    # Create a new column for the block number
    log_data['block_index'] = (log_data['trial_index'] - 1) // block_len + 1

    # Create a new column for trial index within each block
    log_data['trial_index_within_block'] = log_data['trial_index'] % block_len
    log_data['trial_index_within_block'] = log_data['trial_index_within_block'].replace(0, block_len)  # Ensure 0 becomes block_len

    return log_data


def fetch_log(path, directory):
    complete_path = os.path.join(directory, "trial_data", f"{path}.json")
    # print(f"path {complete_path}")
    
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


def bernoulli_entropy(p):
    """
    Computes the entropy of a Bernoulli random variable with probability p.
    """
    if p == 0 or p == 1:
        return 0  # log(0) is undefined, but entropy is 0 in these cases
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def block_entropy(p_x, p_y):
    """
    Computes the joint entropy of two independent Bernoulli random variables.
    """
    return bernoulli_entropy(p_x) + bernoulli_entropy(p_y)

def pupil_histgram_by_stimulus_consistency(log_data, pupil_stat, stats_test = True):
    pupil_max_consistent = []
    pupil_max_inconsistent = []

    for i in range(270):
        # Check the 'stimulus_consistency' condition
        if log_data['stimulus_consistency'].iloc[i] == True:
            pupil_max_consistent.append(pupil_stat[i])
        else:
            pupil_max_inconsistent.append(pupil_stat[i])


    # Create the histogram
    plt.hist(pupil_max_consistent, bins=20, alpha=0.5, label='Consistent', color='blue')
    plt.hist(pupil_max_inconsistent, bins=20, alpha=0.5, label='Inconsistent', color='red')

    # Adding labels and title
    plt.xlabel('Pupil Size')
    plt.ylabel('Frequency')
    plt.title('Pupil Size Histogram by Stimulus Consistency')

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()

    if stats_test:
        # Perform Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(pupil_max_consistent, pupil_max_inconsistent)

        # Print results
        print(f"U-statistic: {u_stat}")
        print(f"P-value: {p_value}")

    return pupil_max_consistent, pupil_max_inconsistent


'''blink processing'''

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

def plot_pupil_size(pupil_data, size=3, figsize = (10,6)):
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




'''functions for calculation surprise'''

def KLdivergence(prior, posterior, epsilon=1e-2):
    prior = np.array(prior)
    posterior = np.array(posterior)
    
    prior = np.where(prior <= 0, epsilon, prior)
    posterior = np.where(posterior <= 0, epsilon, posterior)
    
    divergence = 0
    for i in range(len(prior)):
        divergence += posterior[i] * np.log(posterior[i] / prior[i])
    
    return divergence

def kl_beta(beta1, beta2):
    """Calculate KL divergence between two frozen Beta distributions."""
    # Extract parameters from frozen distributions
    alpha1, beta1_param = beta1.args
    alpha2, beta2_param = beta2.args

    # KL divergence calculation
    term1 = betaln(alpha2, beta2_param) - betaln(alpha1, beta1_param)
    term2 = (alpha1 - alpha2) * digamma(alpha1)
    term3 = (beta1_param - beta2_param) * digamma(beta1_param)
    term4 = (alpha2 - alpha1 + beta2_param - beta1_param) * digamma(alpha1 + beta1_param)
    
    return term1 + term2 + term3 + term4

def kl_dirichlet(alpha1, alpha2):
    """Compute KL divergence between two Dirichlet distributions."""
    alpha1 = np.array(alpha1, dtype=np.float64)
    alpha2 = np.array(alpha2, dtype=np.float64)

    alpha1_0 = np.sum(alpha1)
    alpha2_0 = np.sum(alpha2)

    kl = (
        gammaln(alpha1_0) - np.sum(gammaln(alpha1))
        - (gammaln(alpha2_0) - np.sum(gammaln(alpha2)))
        + np.sum((alpha1 - alpha2) * (digamma(alpha1) - digamma(alpha1_0)))
    )
    return kl

def dirichlet_expectation(alpha):
    # Calculate the sum of the parameters
    alpha_sum = sum(alpha)
    
    # Calculate the expectation for each component
    expectations = [a / alpha_sum for a in alpha]
    
    return expectations

def construct_bern_from_beta(probability):
    return [probability, 1 - probability]

def cue_posterior(cue_direction):
    mapping = {
    'left': [1, 0],
    'right': [0, 1]
    }
    return mapping.get(cue_direction) 

def action_posterior(correct_action):
    mapping = {
    'f': [1, 0],
    'j': [0, 1]
    }
    return mapping.get(correct_action) 


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

def get_cue_learning_surprise(log_data):
    """
    Updates log_data with:
    - 'cue_learning_prior': Prior belief (Beta distribution).
    - 'cue_learning_posterior': Posterior belief (Beta distribution).
    - 'cue_learning_surprise': KL divergence (surprise measure).
    """
    left_count = right_count = 1  # Initial Beta(1, 1) prior
    previous_block_index = 1

    # Ensure log_data has the necessary columns
    log_data['cue_learning_prior'] = None
    log_data['cue_learning_posterior'] = None
    log_data['cue_learning_surprise'] = None

    # Iterate through rows
    for i, row in log_data.iterrows():
        # Check if there's a change in block index:
        current_block_index = row['block_index']
        if current_block_index > previous_block_index:
            previous_block_index = current_block_index
            left_count = right_count = 1  # Reset counts for new block

        # Calculate prior based on the current counts
        prior = beta(left_count, right_count)

        # Store the prior belief for the current row
        log_data.at[i, 'cue_learning_prior'] = prior  
        cue = row['cue_direction']

        # Update counts based on the observed action
        if cue == 'left':
            left_count += 1
        elif cue == 'right':
            right_count += 1

        # Update posterior belief and calculate KL divergence (surprise)
        posterior = beta(left_count, right_count)
        log_data.at[i, 'cue_learning_posterior'] = posterior
        log_data.at[i, 'cue_learning_surprise'] = kl_beta(prior, posterior)

        # Update prior for the next iteration (next trial)
        # prior = posterior

    return log_data


def get_action_learning_surprise(log_data):
    """
    Updates log_data with:
    - 'action_learning_prior': Prior belief (Beta distribution).
    - 'action_learning_posterior': Posterior belief (Beta distribution).
    - 'action_learning_surprise': KL divergence (surprise measure).
    """
    left_count = right_count = 1  # Initial Beta(1, 1) prior
    previous_block_index = 1

    # Ensure log_data has the necessary columns
    log_data['action_learning_prior'] = None
    log_data['action_learning_posterior'] = None
    log_data['action_learning_surprise'] = None

    # Iterate through rows
    for i, row in log_data.iterrows():
        # Check if there's a change in block index:
        current_block_index = row['block_index']
        if current_block_index > previous_block_index:
            previous_block_index = current_block_index
            left_count = right_count = 1  # Reset counts for new block

        # Calculate prior based on the current counts
        prior = beta(left_count, right_count)

        # Store the prior belief for the current row
        log_data.at[i, 'action_learning_prior'] = prior  
        action = row['response']

        # Update counts based on the observed action
        if action == 'f':
            left_count += 1
        elif action == 'j':
            right_count += 1

        # Update posterior belief and calculate KL divergence (surprise)
        posterior = beta(left_count, right_count)
        log_data.at[i, 'action_learning_posterior'] = posterior
        log_data.at[i, 'action_learning_surprise'] = kl_beta(prior, posterior)

        # Update prior for the next iteration (next trial)
        prior = posterior

    return log_data


def get_sensory_learning_surprise(log_data):
    """
    Updates log_data with:
    - 'prior': Prior Dirichlet parameters.
    - 'posterior': Posterior Dirichlet parameters.
    - 'surprise': KL divergence (surprise measure).
    """
    # Initial Dirichlet(1,1,1,1) prior
    counts = np.array([1, 1, 1, 1])  # [AA, AB, BA, BB]
    previous_block_index = 1

    # Ensure log_data has the necessary columns
    log_data['sensory_learning_prior_param'] = None
    log_data['sensory_learning_posterior_param'] = None
    log_data['sensory_learning_surprise'] = None

    # Iterate through rows
    for i, row in log_data.iterrows():
        # Check if there's a change in block index:
        current_block_index = row['block_index']
        if current_block_index > previous_block_index:
            previous_block_index = current_block_index
            counts = np.array([1, 1, 1, 1])  # Reset counts for new block

        # Store the prior belief (Dirichlet parameters)
        prior = counts.copy()
        log_data.at[i, 'sensory_learning_prior_param'] = prior.tolist()

        # Update counts based on the observed stimulus
        stimulus = row['stimulus']
        if stimulus == 'A|A':
            counts[0] += 1
        elif stimulus == 'A|B':
            counts[1] += 1
        elif stimulus == 'B|A':
            counts[2] += 1
        elif stimulus == 'B|B':
            counts[3] += 1

        # Posterior distribution
        posterior = counts.copy()
        log_data.at[i, 'sensory_learning_posterior_param'] = posterior.tolist()

        # Compute KL divergence (surprise)
        surprise = kl_dirichlet(prior, posterior)
        log_data.at[i, 'sensory_learning_surprise'] = surprise

    return log_data


def cueMemory_posterior(cue_prob, stimulus): 
    cue_posterior = construct_bern_from_beta(cue_prob)
    
    if stimulus == 'B|A':
        # Switch the two entries of the list
        cue_posterior = [cue_posterior[1], cue_posterior[0]]
    
    return cue_posterior



def get_cueMemory_surprise(log_data):
    # Initialize the new column with None
    log_data['cueMemory_surprise_prior'] = None

    # Define mapping for direct stimulus conditions
    stimulus_mapping = {
        'A|A': [1, 0],
        'B|B': [0, 1]
    }

    # Apply function to set values efficiently
    # log_data['cueMemory_surprise_prior'] = log_data.apply(
    #     lambda row: stimulus_mapping.get(row['stimulus'], construct_bern_from_beta(row['action_learning_posterior'].mean())),
    #     axis=1
    # )
    log_data['cueMemory_surprise_prior'] = log_data.apply(
        lambda row: stimulus_mapping.get(row['stimulus'], cueMemory_posterior(row['cue_learning_posterior'].mean(),row['stimulus'])),
        axis=1
    )
    log_data['cueMemory_surprise_posterior'] = log_data.apply(lambda row: action_posterior(row['correct_response']), axis=1)
    log_data['cueMemory_surprise'] = log_data.apply(
        lambda row: KLdivergence(row['cueMemory_surprise_prior'], row['cueMemory_surprise_posterior']), axis=1
    )
    return log_data


def get_information_gain(log_data):
    
    get_cue_learning_surprise(log_data)

    log_data['cue_posterior'] = log_data['cue_direction'].apply(lambda x:cue_posterior(x))
    log_data['cue_prior'] = log_data['cue_learning_prior'].apply(lambda x: construct_bern_from_beta(x.mean()))
    log_data['cue_surprise'] = log_data.apply(
        lambda row: KLdivergence(row['cue_prior'], row['cue_posterior']), axis=1
    )

    get_action_learning_surprise(log_data)

    log_data['action_prior'] = log_data['action_learning_prior'].apply(lambda x: construct_bern_from_beta(x.mean()))
    log_data['action_posterior'] = log_data['correct_response'].apply(lambda x:action_posterior(x))
    log_data['action_surprise'] = log_data.apply(
        lambda row: KLdivergence(row['action_prior'], row['action_posterior']), axis=1
    )

    get_sensory_learning_surprise(log_data)

    log_data['sensory_prior'] = log_data['sensory_learning_prior_param'].apply(lambda x: dirichlet_expectation(x))
    log_data['sensory_posterior'] = log_data['stimulus'].apply(lambda x:sensory_posterior(x))
    log_data['sensory_surprise'] = log_data.apply(
        lambda row: KLdivergence(row['sensory_prior'], row['sensory_posterior']), axis=1
    )
    
    get_cueMemory_surprise(log_data)

    return log_data


def select_pupil_statistics(pupil_statistics, statistics: str, event: str):
    
    # Validate inputs
    if statistics not in statistics_mapping:
        raise ValueError(f"Invalid statistic: {statistics}. Valid options are 'max', 'mean', 'median'.")
    if event not in event_mapping:
        raise ValueError(f"Invalid event: {event}. Valid options are 'sound', 'cue', 'response'.")

    # Get the corresponding codes from the dictionaries
    statistics_code = statistics_mapping[statistics]
    _, event_code = event_mapping[event]
    
    # Access the pupil statistics
    pupil_stats = pupil_statistics[:, event_code, statistics_code]
    
    return pupil_stats



'''mappings'''

statistics_mapping = {
    'max': 0,
    'mean': 1,
    'median': 2
}
        
event_mapping = {
    'cue': ('show cue', 0),
    'sound': ('start decision window', 1),
    'response': ('response', 2)
}



def correlation_heatmap(df, title='Correlation Heatmap', figsize=(10, 8), cmap='coolwarm'):
    """
    Plots a correlation heatmap for the numeric columns in the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - title (str): Title for the heatmap.
    - figsize (tuple): Size of the figure (width, height).
    - cmap (str): Color map used for heatmap.
    """
    if 'subject' in df.columns:
        df = df.drop(columns=['subject'])

    corr_matrix = df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def select_predictors(log_data, stage=None, stimulus_consistency = False, cue_surprise_true = False):
    """
    Select predictors from log_data based on the specified stage.

    Parameters:
    - log_data (pd.DataFrame): The input log data containing all predictors.
    - stage (str): The stage of the experiment ('cue', 'sound', or None).

    Returns:
    - predictor (pd.DataFrame): A DataFrame containing the selected predictors.
    """
  

    if stage == 'cue':
        predictor = log_data[['subject', 'cue_learning_surprise', 'cue_surprise']]
    elif stage == 'sound':
        predictor = log_data[['subject', 'action_learning_surprise', 'action_surprise',
                              'sensory_learning_surprise', 'sensory_surprise', 'cueMemory_surprise']]
    elif stage is None:
        predictor = log_data[['cue_learning_surprise', 'cue_surprise',
                              'action_learning_surprise', 'action_surprise',
                              'sensory_learning_surprise', 'sensory_surprise',
                              'cueMemory_surprise']]
    else:
        raise ValueError(f"Invalid stage: {stage}. Choose 'cue', 'sound', or None.")
    
    if stimulus_consistency:
        predictor['stimulus_consistency'] = log_data['stimulus_consistency']


    if cue_surprise_true: 
        
        # Create 'cue_prob' column
        log_data['cue_prob'] = np.where(
            log_data['cue_direction'] == 'left',
            log_data['left_cue_condition'],
            1 - log_data['left_cue_condition']
        )

        # Add 'cue_surprise_true' as the negative log of 'cue_prob'
        log_data['cue_surprise_true'] = -np.log(log_data['cue_prob'])

        predictor['cue_surprise_true'] = log_data['cue_surprise_true']
    
    return predictor



def correlate(log_data, pupil_data_list: list, stage: str, pupil_stats='max', congruency = None, drop_incorrect = True):
    """
    Correlate pupil statistics with predictors in log data using a mixed-effects model.

    Parameters:
    - log_data (pd.DataFrame): DataFrame containing trial-by-trial predictors and subject info.
    - pupil_data_list (list): List of pupil data arrays (one per subject).
    - stage (str): Either 'cue' or 'sound' — determines which predictors to use.
    - pupil_stats (str): Which pupil statistic to select ('max', 'mean', etc.).

    Returns:
    - result (MixedLMResults): The fitted mixed-effects model result.
    """
    
    # Step 1: Extract pupil statistics for each subject
    pupil_stats_list = [select_pupil_statistics(pupildata, pupil_stats, stage) for pupildata in pupil_data_list]
    stacked_pupil_stats = np.hstack(pupil_stats_list)


    # Step 2: Select predictors based on stage
    predictor = select_predictors(log_data, stage)
    if stage == 'cue':
        formula = "stacked_pupil_stats ~ cue_learning_surprise + cue_surprise"
    elif stage == 'sound':
        formula = ("stacked_pupil_stats ~ action_learning_surprise + action_surprise + "
                   "sensory_learning_surprise + sensory_surprise + cueMemory_surprise")
    else:
        raise ValueError(f"Invalid stage: {stage}. Choose 'cue' or 'sound'.")

    print(len(stacked_pupil_stats),len(predictor))


    # Step 3: Make sure predictor data are numeric
    predictor = predictor.apply(pd.to_numeric, errors='coerce')

    # Step 4: Add pupil data into the predictor DataFrame
    predictor['stacked_pupil_stats'] = stacked_pupil_stats

    if congruency == True:
        predictor = predictor[predictor['stimulus_consistency'] == 1]
        print("selected congruent trials")
    
    if congruency == False:
        predictor = predictor[predictor['stimulus_consistency'] == 0]
        print("selected incongruent trials")

    if drop_incorrect: 
        if 'performance' in predictor.columns:
            predictor = predictor[predictor['performance'] == 'correct']

    predictor.dropna(inplace=True)

    # Step 5: Fit the mixed-effects model
    model = smf.mixedlm(
        formula,
        predictor,
        groups=predictor["subject"]
    )
    result = model.fit()

    # Step 6: Print and return results
    print(result.summary())
    
    return result

'''normalization'''


def normalize_by_first_valid(pupil_array, drop1sec = True):
    pupil_array = np.array(pupil_array, dtype=float)  # ensure it's a float array
    valid_values = pupil_array[~np.isnan(pupil_array)]
    baseline = valid_values[0]
    normalized = (pupil_array - baseline) / baseline

    if drop1sec:
        return normalized[1000:]  # drop first 1000 timepoints
    else:
        return normalized
    

def normalize_after_baseline(pupil_array, baseline_len: int = 1000, drop_baseline: bool = True):
    """
    Normalize a single pupil trace by subtracting the mean of the first `baseline_len` points.

    Parameters:
    - pupil_array: list or array of pupil size values
    - baseline_len: number of initial points to compute the baseline (default = 1000)
    - drop_baseline: if True, return data after baseline period; otherwise return full-length data

    Returns:
    - normalized: np.ndarray with baseline-subtracted values
    """
    pupil_array = np.asarray(pupil_array, dtype=float)

    if len(pupil_array) <= baseline_len:
        return np.full_like(pupil_array, np.nan)  # Not enough data

    baseline = np.nanmean(pupil_array[:baseline_len])
    normalized = pupil_array - baseline

    if drop_baseline:
        return normalized[baseline_len:]
    else:
        return normalized


# def normalize_after_baseline(data: List[List[float]], baseline_len: int = 1000) -> Tuple[List[np.ndarray], List[float]]:
#         """
#         Normalize time series data by subtracting the baseline mean.

#         For each list in the input:
#         - Use the first `baseline_len` points to compute a baseline mean.
#         - Subtract that baseline from the rest of the data (after baseline).
#         - Skip entries shorter than or equal to `baseline_len`.

#         Parameters:
#         - data: List of lists (or arrays), each representing one time series.
#         - baseline_len: Number of initial points to use for baseline calculation (default = 1000).

#         Returns:
#         - normalized_data: List of NumPy arrays with baseline-subtracted values (post-baseline only).
#         - baselines: List of computed baseline means (one per valid series).
#         """
#         normalized_data = []
#         baselines = []

#         for lst in data:
#             lst = np.asarray(lst)
#             if len(lst) <= baseline_len:
#                 continue  # Skip entries too short to extract a baseline

#             baseline = np.nanmean(lst[:baseline_len])
#             normalized_segment = lst[baseline_len:] - baseline
#             normalized_data.append(normalized_segment)
#             baselines.append(baseline)

#         return normalized_data, baselines


def average_by_position(data):
    max_len = max(len(lst) for lst in data)
    averages = []

    for i in range(max_len):
        values_at_i = [lst[i] for lst in data if len(lst) > i and not np.isnan(lst[i])]
        if values_at_i:
            avg = np.mean(values_at_i)
        else:
            avg = np.nan  # Or 0, or skip appending, depending on what you want
        averages.append(avg)

    return averages

def mean_and_std_by_position(data):
    max_len = max(len(lst) for lst in data)
    means = []
    stds = []

    for i in range(max_len):
        values_at_i = [lst[i] for lst in data if len(lst) > i and not np.isnan(lst[i])]
        if values_at_i:
            means.append(np.mean(values_at_i))
            stds.append(np.std(values_at_i))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    
    return np.array(means), np.array(stds)

def pad_lists_to_same_length(list_of_lists, fill_value=np.nan):
    max_length = max(len(lst) for lst in list_of_lists)
    padded = []
    for lst in list_of_lists:
        lst = list(lst)  # make sure it's a list
        pad_len = max_length - len(lst)
        padded.append(lst + [fill_value] * pad_len)
    return np.array(padded)



def analyze_pupil_data_by_stimulus_consistency(pilot, normalize_func = normalize_after_baseline, pad_func= pad_lists_to_same_length):
    # Compute pupil max and initial histogram (optional)
    pupil_max = pilot.select_pupil_statistics('max', 'sound')
    pupil_histgram_by_stimulus_consistency(pilot.log_data, pupil_max)

    # Normalize pupil data
    normalized_pupils_alltrials = [
        normalize_func(pupil_trace) for pupil_trace in pilot.rawPupil_sound
    ]
    normalized_pupils_alltrials = np.array(normalized_pupils_alltrials, dtype=object)

    '''return normalized_pupils_alltrials '''

    '''return max mean medium after normalization '''

    '''separate the bottom part into independent function'''
    # Overall mean and std
    mean_pattern, std_pattern = mean_and_std_by_position(normalized_pupils_alltrials)

    # Split data by stimulus_consistency
    true_indices = pilot.log_data[pilot.log_data['stimulus_consistency'] == True].index
    false_indices = pilot.log_data[pilot.log_data['stimulus_consistency'] == False].index
    selected_data_true = normalized_pupils_alltrials[true_indices]
    selected_data_false = normalized_pupils_alltrials[false_indices]

    def compute_mean_std(data):
        mean_list, std_list = [], []
        for i in range(data.shape[1]):
            try:
                values = np.array(data[:, i], dtype=float)
                valid = values[~np.isnan(values)]
            except ValueError:
                valid = np.array([])
            mean_list.append(np.mean(valid) if valid.size > 0 else np.nan)
            std_list.append(np.std(valid) if valid.size > 0 else np.nan)
        return np.array(mean_list), np.array(std_list)

    mean_pattern_true, std_pattern_true = compute_mean_std(selected_data_true)
    mean_pattern_false, std_pattern_false = compute_mean_std(selected_data_false)

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(24, 4), sharey=True)

    axs[0].errorbar(np.arange(len(mean_pattern)), mean_pattern, yerr=std_pattern, fmt='-o', color='mediumblue', ecolor='lightblue', capsize=3)
    axs[0].set_title("Overall Mean")
    axs[0].set_xlabel("Time Index")
    axs[0].set_ylabel("Pupil Size")
    axs[0].grid(True)

    axs[1].errorbar(np.arange(len(mean_pattern_true)), mean_pattern_true, yerr=std_pattern_true, fmt='-o', color='dodgerblue', ecolor='lightblue', capsize=3)
    axs[1].set_title("True Indices")
    axs[1].set_xlabel("Time Index")
    axs[1].grid(True)

    axs[2].errorbar(np.arange(len(mean_pattern_false)), mean_pattern_false, yerr=std_pattern_false, fmt='-o', color='tomato', ecolor='lightcoral', capsize=3)
    axs[2].set_title("False Indices")
    axs[2].set_xlabel("Time Index")
    axs[2].grid(True)

    axs[3].plot(mean_pattern_true, label="True", color='dodgerblue', marker='o', markersize=4)
    axs[3].plot(mean_pattern_false, label="False", color='tomato', marker='o', markersize=4)
    axs[3].set_title("True vs. False")
    axs[3].set_xlabel("Time Index")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

    # Pad for statistical testing
    padded_true = pad_func(selected_data_true)
    padded_false = pad_func(selected_data_false)

    # Timepoint-wise t-tests
    results = []
    for i in range(1000, 2002):
        if i >= padded_true.shape[1] or i >= padded_false.shape[1]:
            continue
        data_true = padded_true[:, i]
        data_false = padded_false[:, i]
        data_true = data_true[~np.isnan(data_true)]
        data_false = data_false[~np.isnan(data_false)]
        if len(data_true) > 1 and len(data_false) > 1:
            stat, pval = ttest_ind(data_true, data_false, equal_var=False)
            results.append((i, stat, pval))
        else:
            results.append((i, np.nan, np.nan))

    results_df = pd.DataFrame(results, columns=["Time Index", "t-statistic", "p-value"])

    # Histogram of p-values
    plt.figure(figsize=(8, 4))
    plt.hist(results_df['p-value'].dropna(), bins=30, color='purple', edgecolor='black')
    plt.title("Histogram of P-values from Timepoint-wise T-tests")
    plt.xlabel("P-value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return normalized_pupils_alltrials, results_df


class DataAnalyze_new:
    def __init__(self, file_name, directory) -> None:
        self.file_name = file_name
    
        self.directory = directory 

        allad_path = os.path.join(self.directory, "eyetracking", file_name, "all_adin.csv")
        all_path = os.path.join(self.directory, "eyetracking", file_name, "all.csv")


        if os.path.exists(allad_path):
            self.eye_raw = pd.read_csv(allad_path)
        else:
            _eye_data = pd.read_csv(all_path)
            self.eye_raw = adjust_trial_index(_eye_data)
            self.eye_raw.to_csv(allad_path, index=False)
            os.remove(all_path)

        # if self.eye_raw is not None and not self.eye_raw.empty:
        #     # print("Eye raw data obtained")
        # else:
        #     print("Eye raw data not obtained or is empty")

        self.log_data = fetch_log(file_name,self.directory)
        # if self.log_data is not None and not self.log_data.empty:
        #     print("Log data obtained")
        #     # print(f"DataFrame columns: {self.log_data.columns.tolist()}")
        #     # print(f"DataFrame shape: {self.log_data.shape}")
        #     # print(f"First few rows:\n{self.log_data.head()}")
        # else:
        #     print("Log data not obtained or is empty")

        # Calculate number of trials and blocks
        self.num_trials = len(self.log_data['trial_index'].unique())
        self.num_blocks = math.ceil(self.num_trials / 30)

        self.log_data = add_block_columns(self.log_data)
        self.log_data['stimulus_consistency'] = self.log_data['stimulus'].isin(['A|A', 'B|B'])

        # # Load pupil statistics
        # try:
        #     stats_path = os.path.join(self.directory, 'eyetracking', file_name, "pupil_statistics_raw.npz")
        #     data = np.load(stats_path)
        #     self.pupil_statistics_raw = data['arr_0']
        #     print("sound stats obtained")
        # except Exception as e:
        #     print(f"Error loading data : {e}")
        #     self.pupil_statistics_raw = np.full((self.num_trials, 3, 3), 0)  #3statistics types, 3 event type

        # # Load raw pupil sound data
        # try:
        #     sound_path = os.path.join(self.directory, 'eyetracking', file_name, "rawPupil_sound.npy")
        #     print("sound path", sound_path)
        #     data = np.load(sound_path, allow_pickle=True)
        #     self.rawPupil_sound = np.array(data, dtype=object)
        #     print("raw sound pupil obtained")
        # except Exception as e:
        #     print("Error loading data rawPupil_sound.npy")
        #     self.get_pupil_raw_sound()

        # # Load raw pupil cue data
        # try:
        #     cue_path = os.path.join(self.directory, 'eyetracking', file_name, "rawPupil_cue.npy")
        #     data = np.load(cue_path, allow_pickle=True)
        #     self.rawPupil_cue = np.array(data, dtype=object)
        #     print("raw cue pupil obtained")
        # except Exception as e:
        #     print("Error loading data rawPupil_cue.npy")
        #     self.get_pupil_raw_cue()

        # Compute trial entropy
        self.log_data['block_entropy'] = self.log_data.apply(
            lambda row: block_entropy(row['left_cue_condition'], row['stimA_condition']), axis=1
        )

        # Initialize trial parameters
        self.trial_parameters = []
        for i in range(self.num_blocks):
            idx = i * 30
            block_sample = self.log_data.iloc[idx]
            left_cue_conditions = block_sample['left_cue_condition']
            stimA_conditions = block_sample['stimA_condition']
            self.trial_parameters.append((left_cue_conditions, stimA_conditions))

        # Compute block entropy
        # self.block_entropy = [self.log_data['trial_entropy'].iloc[i * 30] for i in range(self.num_blocks)]

        # # Initialize placeholders for normalized pupil data
        # self.normalized_pupil_sound = np.full((self.num_trials, 3), np.nan)
        # # self.baseline_sound = []
        # self.normalized_pupil_cue = np.full((self.num_trials, 3), np.nan)
        # # self.baseline_cue = []


        

    def show_performance(self):
        '''incorrect trial'''
        self.incorrect_count = (self.log_data["performance"] == "Incorrect").sum()
        self.incorrect_trial_list = self.log_data.loc[self.log_data["performance"] == "Incorrect", "trial_index"]
        print("incorrect count:", self.incorrect_count)
        print("incorrect trial list:", self.incorrect_trial_list.values)
        # Assuming self.num_block and self.incorrect_trial_list are already defined
        print("Number of incorrect trials per block:")

        # Initialize a list to store the count of incorrect trials per block
        incorrect_trials_per_block = []

        # Iterate through each block and count the trials within the range [i*30, (i+1)*30)
        for i in range(self.num_blocks):
            # Define the start and end of the block range
            block_start = i * 30
            block_end = (i + 1) * 30
            
            # Count how many trials in incorrect_trial_list fall within this range
            count_in_block = self.incorrect_trial_list.between(block_start, block_end - 1).sum()
            
            # Append the count to the list
            incorrect_trials_per_block.append(count_in_block)
            
            # Print the count for each block
            print(f"Block {i}:{self.trial_parameters[i]}, {count_in_block} incorrect trials")

        # Optionally, store the result in self for later use
        self.incorrect_trials_per_block = incorrect_trials_per_block


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


    def get_block_data(self, block_index):
        block_data = self.log_data[self.log_data['block'] == block_index]
        return block_data
    

    def visualize_raw_processed_pair(self, eye_data=None, trial_index=None):
        if not isinstance(trial_index, list):
            trial_index = [trial_index]
        
        # print(trial_index,"trial_index")

        try:
            # Corrected condition check for 'trial_index'
            if trial_index is None:  
                trial_index = eye_data['trial_index'].unique().tolist()

            # Corrected condition check for 'eye_data'
            if eye_data is None:
                _eye_data = self.eye_raw[self.eye_raw['trial_index']==trial_index[0]]
            else:
                _eye_data = eye_data  # Use provided eye_data directly if available
            
            # Fixed syntax for .dropna() and improved logic
            pupil_data = _eye_data[_eye_data['Pupil'].notna()]  
            # print(pupil_data) 
            # print('trial index:', trial_index, ";trial index type:", type(trial_index))
            filtered_pupil_data = self.filter_eye_data_diff_blink(pupil_data, trial_index)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 3 rows, 1 column of subplots

            # First subplot
            axes[0].scatter(pupil_data['TimeEvent'], pupil_data['Pupil'], color='blue', alpha=0.6, s=3)
            axes[0].set_title(f"Processed Trial Data - Trial {pupil_data['trial_index'].iloc[0]}")
            axes[0].set_xlabel('TimeEvent')
            axes[0].set_ylabel('Pupil Size')
            axes[0].grid(True)

            # Second subplot
            axes[1].scatter(filtered_pupil_data['TimeEvent'], filtered_pupil_data['Pupil'], color='green', alpha=0.6, s=3)
            axes[1].set_title(f"Processed Event Data - Trial {filtered_pupil_data['trial_index'].iloc[0]}")
            axes[1].set_xlabel('TimeEvent')
            axes[1].set_ylabel('Pupil Size')
            axes[1].grid(True)

            # Adjust layout for better spacing
            plt.tight_layout()

            # Show the plot
            plt.show()
        except IndexError:
            raise IndexError("Please specify a valid trial index")


    def visualize_all(self,specify_range: tuple = None):
        if specify_range:
            start, end = specify_range  # Correct unpacking
        else:
            start = 1
            end = self.num_trials +1
        for i in range(start,end):
            # print(i)
            self.visualize_raw_processed_pair(trial_index=i)



    def filter_eye_data_diff_blink(self, eye_data, trial_index_list, rep = 3, dot_size = 4, diff_lower_bound=-2, diff_upper_bound=2, alpha_inner=0.5, alpha_outer = 0.5, inner_bound=0.2, outer_bound=0.22, plot = False):

        eye_data = eye_data[eye_data['trial_index'].isin(trial_index_list)].copy()

        blink_data = record_blink(eye_data)
        #print(blink_data)


        for rep in range(rep):
            print(f'rep number {rep}')
    
            #find reasonable pupil size bound
            Avg_Pupil_Before, Avg_Pupil_After = find_pupil_bounds_around_blink(eye_data, blink_data, inner_bound=inner_bound, outer_bound=outer_bound)
            # print("Pupil bounds before blink: ", Avg_Pupil_Before, "Pupil bounds after blink: ", Avg_Pupil_After)
            blink_data['Avg_Pupil_Before'] = Avg_Pupil_Before
            blink_data['Avg_Pupil_After'] = Avg_Pupil_After

            # exclude unreasonable pupil size
            eye_data = select_valid_pupil_size(eye_data, blink_data)
            print("row numbers after blink process: ", len(eye_data))
            
            #exclude unreasonable differences
            eye_data = calculate_pupil_differences(eye_data)

            eye_data = eye_data[
                (eye_data['Pupil_diff'] >= diff_lower_bound) &
                (eye_data['Pupil_diff'] <= diff_upper_bound) 
            ]

            print("row number after diff filtering: ", len(eye_data))
            # plot_pupil_size(eye_data)
            
            #print("inner bound:", inner_bound,"outer bound", outer_bound)

            if rep == 0: 
                inner_bound=alpha_inner*inner_bound
                outer_bound=alpha_outer*outer_bound

        
        if plot:
            plot_pupil_size(eye_data, size=dot_size)

        return eye_data

    
    def fillin_pupil_statistics(self, trial_index, event, max, mean, median):
        _, event_code = event_mapping[event]
        self.pupil_statistics_raw[trial_index-1,event_code ] = [max, mean, median]
    

    def get_pupil_statistics_raw(self):
        
        # Loop through each trial
        for trial_index in range(1, self.num_trials + 1):

            # Get the data for the current trial, ensuring that Pupil values are not NaN
            _trial_data = self.eye_raw[self.eye_raw['trial_index'] == trial_index].copy()
            
            # List of events we care about ('cue', 'sound', and 'response')
            events = ['cue', 'sound', 'response']
            for event in events:
                event_label, event_code= event_mapping[event]

                if np.all(~np.isnan(self.pupil_statistics_raw[trial_index-1, event_code, :])):
                    continue
                else:
                    print("processing trial ", trial_index, event, flush=True)
                    # if trial_index == 29:
                    #     print(_trial_data[_trial_data['event'] == label].dropna(subset=['TimeEvent']))
                # Try to get the event time for the current event in the trial

                event_time = _trial_data[_trial_data['Event'] == event_label].dropna(subset=['TimeEvent'])['TimeEvent'].iloc[0]
                
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
                    self.visualize_raw_processed_pair(trial_index=[trial_index])
                    self.fillin_pupil_statistics(trial_index, event, np.nan, np.nan, np.nan)
                    print(f"processed_event_data is empty for trial {trial_index} and event {event} (event time: {event_time}).Filled in with nan instead.")

                    # raise ValueError(f"Error: processed_event_data is empty for trial {trial_index} and event {event} (event time: {event_time}). Check filtering conditions.")

                # Calculate the pupil statistics for the current event
                pupil_max = processed_event_data['Pupil'].max()  # Max pupil size in the time window
                pupil_mean = processed_event_data['Pupil'].mean()  # Mean pupil size in the time window
                pupil_median = processed_event_data['Pupil'].median()  # median pupil size in the time window

                self.fillin_pupil_statistics(trial_index, event, pupil_max, pupil_mean, pupil_median)

        print("directory, file name", self.directory, self.file_name)

        path = os.path.join(self.directory, 'eyetracking', self.file_name, "pupil_statistics_raw.npz")
        print(f"Saving pupil statistics to: {path}")
        # Save the pupil statistics to a .npz file
        np.savez(path, self.pupil_statistics_raw)


        return self.pupil_statistics_raw


    def get_pupil_statistics_normalized(self, event=None, normalize_func=normalize_by_first_valid):
        """
        Normalize pupil data and compute max, mean, and median for each trial
        from timepoint 500 to 2500. Results are stored in the corresponding attribute.

        Parameters:
        - event: 'cue', 'sound', or None (processes both if None)
        - normalize_func: function to normalize each pupil trace
        """
        events_to_process = ['cue', 'sound'] if event is None else [event]

        for evt in events_to_process:
            try:
                if evt == 'cue':
                    pupil_data = self.rawPupil_cue
                    pupil_statistics_attr = 'normalized_pupil_cue'
                elif evt == 'sound':
                    pupil_data = self.rawPupil_sound
                    pupil_statistics_attr = 'normalized_pupil_sound'
                else:
                    raise ValueError("Invalid event type. Use 'cue' or 'sound'.")
            except AttributeError:
                print(f"Pupil data for '{evt}' not found — please calculate the raw data first.")
                continue

            # Initialize the array if not already
            existing_data = getattr(self, pupil_statistics_attr, None)
            if existing_data is None or len(existing_data) == 0:
                num_trials = len(pupil_data)
                pupil_statistics_normalized = np.full((num_trials, 3), np.nan)
                setattr(self, pupil_statistics_attr, pupil_statistics_normalized)
            else:
                pupil_statistics_normalized = existing_data

            # Normalize and compute statistics
            for trial_idx, trace in enumerate(pupil_data):
                trace_array = np.array(trace, dtype=float)

                if np.all(np.isnan(trace_array)):
                    pupil_statistics_normalized[trial_idx, :] = np.nan
                    print(f"Trial {trial_idx} ({evt}) is all NaN — skipping.")
                    continue

                # Proceed only if valid
                trace_array = normalize_func(trace_array)
                trace_array = np.array(trace_array, dtype=float)

                if trace_array.size > 2500:
                    window = trace_array[500:2501]
                elif trace_array.size > 500:
                    window = trace_array[500:]
                else:
                    window = np.array([])

                valid = window[~np.isnan(window)]

                pupil_statistics_normalized[trial_idx, 0] = np.max(valid) if valid.size > 0 else np.nan
                pupil_statistics_normalized[trial_idx, 1] = np.mean(valid) if valid.size > 0 else np.nan
                pupil_statistics_normalized[trial_idx, 2] = np.median(valid) if valid.size > 0 else np.nan

                print(f"Filled statistics for trial {trial_idx} ({evt})")



    def select_pupil_statistics(self, statistics: str, event: str):
        return select_pupil_statistics(self.pupil_statistics_raw, statistics, event)

    def select_pupil_statistics_normalized(self, statistics: str, event: str):
        """
        Select a specific type of normalized pupil statistic for a given event.

        Parameters:
        - statistics: one of 'max', 'mean', or 'median'
        - event: one of 'cue' or 'sound'

        Returns:
        - 1D NumPy array of the selected statistic across trials
        """
        # Validate inputs first
        if statistics not in statistics_mapping:
            raise ValueError(f"Invalid statistic: '{statistics}'. Valid options are: {list(statistics_mapping.keys())}.")
        if event not in event_mapping:
            raise ValueError(f"Invalid event: '{event}'. Valid options are: {list(event_mapping.keys())}.")

        # Get codes from mappings
        statistics_code = statistics_mapping[statistics]
        
        # Select the appropriate pupil statistics array
        if event == 'cue':
            pupil_stats = self.normalized_pupil_cue
        elif event == 'sound':
            pupil_stats = self.normalized_pupil_sound

        # Access and return the selected statistic column
        return pupil_stats[:, statistics_code]



    def get_pupil_raw_sound(self) -> None:
        """
        Extract and align pupil data from each trial based on sound onset.
        Saves the resulting array (shape: 270 x 6000) as rawPupil_sound.npy.
        """
        pupils_alltrials = []

        for i in range(1, 271):
            try:
                if i < 270:
                    trials = [i, i + 1]
                else:
                    trials = [i]

                neighbor_trial_data = self.eye_raw[self.eye_raw['trial_index'].isin(trials)].copy()
                message_data = neighbor_trial_data[neighbor_trial_data['Type'] == 'Message']
                pupil_data = self.filter_eye_data_diff_blink(neighbor_trial_data, trials)

                if i < 270:
                    sound_time = message_data[
                        (message_data['trial_index'] == i) & 
                        (message_data['Event'] == 'start sound')
                    ]['TimeEvent'].iloc[0]

                    nextCue_time = message_data[
                        (message_data['trial_index'] == i + 1) & 
                        (message_data['Event'] == 'show cue')
                    ]['TimeEvent'].iloc[0]

                    dt = nextCue_time - sound_time
                else:
                    sound_time = message_data[
                        message_data['Event'] == 'start sound'
                    ]['TimeEvent'].iloc[0]
                    dt = 10

                # Time range logic
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

                # De-duplicate based on rounded index
                rounded_indices = rounded_indices[~rounded_indices.duplicated(keep='first')]

                pupils_1trial = np.full(6000, np.nan)
                for idx, rounded_idx in zip(rounded_indices.index, rounded_indices.values):
                    pupil_size = filtered_pupil_data.loc[idx, 'Pupil']
                    if 0 < rounded_idx <= 6000:
                        pupils_1trial[rounded_idx - 1] = pupil_size

                pupils_alltrials.append(pupils_1trial)
                print(f"Done with trial sound {i} in file {self.file_name}")

            except Exception as e:
                print(f"Error in trial {i}, folder {self.file_name}: {e}")
                continue

        save_path = os.path.join(self.directory, self.file_name, "rawPupil_sound.npy")
        np.save(save_path, np.array(pupils_alltrials, dtype=object))
        print(f"Saved data for folder {self.file_name}")

    def get_pupil_raw_cue(self) -> None:
        """
        Align pupil data to cue onset and save each trial’s 3-second window.
        Output shape: (270, 3000), each row = 3 seconds at 1000Hz.
        """
        pupils_alltrials = []

        for i in range(1, 271):
            try:
                if i == 1:
                    trials = [i]
                else:
                    trials = [i - 1, i]

                neighbor_trial_data = self.eye_raw[self.eye_raw['trial_index'].isin(trials)].copy()
                message_data = neighbor_trial_data[neighbor_trial_data['Type'] == 'Message']
                pupil_data = self.filter_eye_data_diff_blink(neighbor_trial_data, trials)

                cue_time = message_data[
                    (message_data['trial_index'] == i) &
                    (message_data['Event'] == 'show cue')
                ]['TimeEvent'].iloc[0]

                filtered_pupil_data = pupil_data[
                    (pupil_data['TimeEvent'] >= cue_time - 1) &
                    (pupil_data['TimeEvent'] <= cue_time + 2)
                ]

                time_data = filtered_pupil_data['TimeEvent'].copy()
                rounded_indices = np.ceil((time_data - (cue_time - 1)) / 0.001).astype(int)

                # Remove duplicates based on index
                rounded_indices = rounded_indices[~rounded_indices.duplicated(keep='first')]

                pupils_1trial = np.full(3000, np.nan)
                for idx, rounded_idx in zip(rounded_indices.index, rounded_indices.values):
                    pupil_size = filtered_pupil_data.loc[idx, 'Pupil']
                    if 0 < rounded_idx <= 3000:
                        pupils_1trial[rounded_idx - 1] = pupil_size

                pupils_alltrials.append(pupils_1trial)
                print(f"Done with trial cue {i} in file {self.file_name}")

            except Exception as e:
                print(f"Error in trial {i}, folder {self.file_name}: {e}")
                continue

        save_path = os.path.join(self.directory, 'eyetracking', self.file_name, "rawPupil_cue.npy")
        np.save(save_path, np.array(pupils_alltrials, dtype=object))
        print(f"Saved cue-locked pupil data for folder {self.file_name}")




    def normalize_pupilStats_by_event(self, event: str) -> np.ndarray:
        """
        Extracts and baseline-corrects the specified event row from a (n_trials, 3, 3) array.
        
        Returns a (n_trials, 3) array with baseline-subtracted values for that event.
        
        Parameters:
        - data: np.ndarray of shape (n_trials, 3, 3)
        - baselines: list of float (length n_trials)
        - event: one of "cue", "sound", or "response"

        Returns:
        - adjusted: np.ndarray of shape (n_trials, 3)
        """
        event_mapping = {"cue": 0, "sound": 1, "response": 2}

        if event not in event_mapping:
            raise ValueError(f"Invalid event '{event}'. Choose from 'cue', 'sound', or 'response'.")
        
        if event =='cue':
            baselines = self.baseline_cue
        elif event == 'sound':
            baselines = self.baseline_sound

        
        event_idx = event_mapping.get(event)

        if event_idx is None:
            raise ValueError(f"Invalid event name: {event}")


        n_trials = self.pupil_statistics_raw.shape[0]
        adjusted = np.full((n_trials, 3), np.nan)

        for i in range(n_trials):
            row = self.pupil_statistics_raw[i, event_idx, :]
            if not np.isnan(row).all():
                mask = ~np.isnan(row)
                adjusted[i, mask] = row[mask] - baselines[i]

        return adjusted
    
    def get_normalized_stats(self):
        self.normalized_pupil_cue, self.baseline_cue = self.normalize_after_baseline(self.rawPupil_cue)
        self.normalize_pupilStats_by_event('cue')

        self.normalized_pupil_sound, self.baseline_sound = self.normalize_after_baseline(self.rawPupil_sound)
        self.normalize_pupilStats_by_event('sound')


    def keep_record(self):
        pupil_data_dict = {
            "pupil_statistics_raw": self.pupil_statistics_raw.tolist(),
            "rawPupil_sound": convert_nested_ndarray_to_list(self.rawPupil_sound),
            "rawPupil_cue": convert_nested_ndarray_to_list(self.rawPupil_cue),
            "normalized_pupil_sound": convert_nested_ndarray_to_list(self.normalized_pupil_sound),
            "normalized_pupil_cue": convert_nested_ndarray_to_list(self.normalized_pupil_cue),
            "baseline_cue": self.baseline_cue,
            "baseline_sound": self.baseline_sound
        }
        file_path = os.path.join(self.directory, self.file_name, "full_record.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write to JSON
        with open(file_path, "w") as f:
            json.dump(pupil_data_dict, f)

