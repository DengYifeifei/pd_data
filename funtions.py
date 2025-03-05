def process_pupil_data(eye_data, trial_index_list, inner_bound, outer_bound, lower_bound=-2, upper_bound=2, window_size=120):
    # Filter data by trial index
    eyedatacopy = eye_data[eye_data['trial_index'].isin(trial_index_list)].copy()
    
    # Find blinks
    blink_data = eyedatacopy[eyedatacopy['Type'] == 'Blink'][['Start', 'End']]
    
    before_pupil_sizes = []
    after_pupil_sizes = []
    
    # Iterate over each blink
    for _, blink in blink_data.iterrows():
        start_time = blink['Start']
        end_time = blink['End']
        
        # Get pupil size before start
        before_data = eyedatacopy[(eyedatacopy['TimeEvent'] >= start_time - outer_bound) &
                                  (eyedatacopy['TimeEvent'] < start_time - inner_bound)]
        avg_before = before_data['Pupil'].mean()
        
        # Get pupil size after end
        after_data = eyedatacopy[(eyedatacopy['TimeEvent'] > end_time + inner_bound) &
                                 (eyedatacopy['TimeEvent'] <= end_time + outer_bound)]
        avg_after = after_data['Pupil'].mean()
        
        before_pupil_sizes.append(avg_before)
        after_pupil_sizes.append(avg_after)
    
    # Add results to blink_data
    blink_data['Avg_Pupil_Before'] = before_pupil_sizes
    blink_data['Avg_Pupil_After'] = after_pupil_sizes
    
    # Filter out invalid pupil sizes
    filtered_data = []
    for _, row in eyedatacopy.iterrows():
        valid = True
        for _, blink in blink_data.iterrows():
            if blink['Start'] - 0.05 <= row['TimeEvent'] <= blink['End'] + 0.05:
                if not (blink['Avg_Pupil_Before'] <= row['Pupil'] <= blink['Avg_Pupil_After']):
                    valid = False
                    break
        if valid:
            filtered_data.append(row)
    
    # Convert filtered_data back to a DataFrame
    filtered_df = pd.DataFrame(filtered_data)
    filtered_df['Pupil_diff'] = filtered_df['Pupil'].shift(-1) - filtered_df['Pupil']
    
    # Apply pupil difference filtering
    filtered_data = filtered_df[
        (filtered_df['Pupil_diff'] >= lower_bound) &
        (filtered_df['Pupil_diff'] <= upper_bound)
    ]
    
    # Compute rolling median and IQR
    rolling_median = filtered_data['Pupil'].rolling(window=window_size, center=True).median()
    rolling_iqr = (filtered_data['Pupil'].rolling(window=window_size, center=True).quantile(0.55) -
                   filtered_data['Pupil'].rolling(window=window_size, center=True).quantile(0.40))
    
    # Define outliers and remove them
    lower_outlier_bound = rolling_median - 1.5 * rolling_iqr
    upper_outlier_bound = rolling_median + 1.5 * rolling_iqr
    
    filtered_data = filtered_data[
        (filtered_data['Pupil'] >= lower_outlier_bound) & 
        (filtered_data['Pupil'] <= upper_outlier_bound)
    ]
    
    # Plot results
    if 'TimeEvent' in filtered_data.columns and 'Pupil' in filtered_data.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_data['TimeEvent'], filtered_data['Pupil'], color='blue', alpha=0.6, label='Pupil Size')
        plt.title('Pupil Size Over TimeEvent')
        plt.xlabel('TimeEvent')
        plt.ylabel('Pupil Size')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Error: filtered_df does not contain 'TimeEvent' or 'Pupil' columns.")
    
    return filtered_data
