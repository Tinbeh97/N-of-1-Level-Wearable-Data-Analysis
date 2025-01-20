def compute_cohens_d(group1, group2):
    """
    Calculate Cohen's d for two independent samples.
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    # Cohen's d
    cohen_d = (mean1 - mean2) / pooled_std
    return cohen_d

def perform_ttest_with_effect_size(feat_name='deep'):
    """
    Perform t-tests for events and compute Cohen's d as an effect size.
    """
    ttest_results = []

    for event in event_type:
        for testUser in df_user_w_events['record_id'].unique():
            user_data = df_user_w_events[df_user_w_events['record_id'] == testUser]
            event_instances = user_data[user_data[event].notna()]
            if event_instances.empty:
                continue
            for _, event_row in event_instances.iterrows():
                event_day = event_row['daysSince']
                period_before = user_data[(user_data['daysSince'] >= event_day - 16) & (user_data['daysSince'] < event_day - 4)]
                period_around = user_data[(user_data['daysSince'] >= event_day - 4) & (user_data['daysSince'] <= event_day + 4)]

                if not period_before.empty and not period_around.empty:
                    if len(period_before) < 7 or len(period_around) < 6:
                        continue

                    deep_before = period_before[feat_name].dropna()
                    deep_around = period_around[feat_name].dropna()

                    if len(deep_before) > 1 and len(deep_around) > 1:
                        t_stat, p_val = ttest_ind(deep_before, deep_around, equal_var=False)
                        cohen_d = compute_cohens_d(deep_before, deep_around)

                        event_name = event_row['AE'] if event == 'AE' else event
                        ttest_results.append({
                            'record_id': testUser,
                            'daysSince': event_day,
                            'event_type': event,
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'cohens_d': cohen_d,
                            'event_name': event_name
                        })

    ttest_results_df = pd.DataFrame(ttest_results)
    return ttest_results_df

def visualize_effect_sizes(ttest_results_df):
    """
    Visualize the distribution of Cohen's d (effect sizes) and its relationship with other variables.
    """
    # 1. Histogram of Cohen's d
    plt.figure(figsize=(8, 6))
    sns.histplot(ttest_results_df['cohens_d'], bins=30, kde=True)
    plt.axvline(0, color='red', linestyle='--', label='No Effect (d = 0)')
    plt.axvline(0.2, color='blue', linestyle='--', label='Small Effect (d = 0.2)')
    plt.axvline(0.5, color='green', linestyle='--', label='Medium Effect (d = 0.5)')
    plt.axvline(0.8, color='purple', linestyle='--', label='Large Effect (d = 0.8)')
    plt.title('Histogram of Effect Sizes (Cohen\'s d)')
    plt.xlabel('Cohen\'s d')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # 2. Boxplot of Cohen's d by event type
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=ttest_results_df, x='event_type', y='cohens_d')
    plt.title('Distribution of Effect Sizes by Event Type')
    plt.xlabel('Event Type')
    plt.ylabel('Cohen\'s d')
    plt.xticks(rotation=45)
    plt.axhline(0, color='red', linestyle='--', label='No Effect (d = 0)')
    plt.legend()
    plt.show()

    # 3. Scatter plot of Cohen's d vs. t-statistic
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=ttest_results_df, x='t_statistic', y='cohens_d', hue='event_type', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', label='No Effect (d = 0)')
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Scatter Plot of Effect Sizes vs. t-Statistic')
    plt.xlabel('t-Statistic')
    plt.ylabel('Cohen\'s d')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    
    # 3. Scatter plot of Cohen's d vs. t-statistic
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=ttest_results_df, x='p_value', y='cohens_d', hue='event_type', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', label='No Effect (d = 0)')
    plt.axvline(0.05, color='red', linestyle='--', label='not significant')
    plt.title('Scatter Plot of Effect Sizes vs. p value')
    plt.xlabel('p value')
    plt.ylabel('Cohen\'s d')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

    # 4. Scatter plot of Cohen's d vs. daysSince
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=ttest_results_df, x='daysSince', y='cohens_d', hue='event_type', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', label='No Effect (d = 0)')
    plt.title('Scatter Plot of Effect Sizes vs. Event Timing (daysSince)')
    plt.xlabel('Days Since Delivery')
    plt.ylabel('Cohen\'s d')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    
def perform_ttest_on_events(feat_name='deep'):
    # Create an empty DataFrame to store the t-test results
    #ttest_results = pd.DataFrame(columns=['record_id', 'daysSince', 'event_type', 't_statistic', 'p_value'])
    ttest_results = []
    # Loop through each event type to perform t-test analysis
    for event in event_type:
        event_name = event
        for testUser in df_user_w_events['record_id'].unique():
            # Extract rows for the current user and event
            user_data = df_user_w_events[df_user_w_events['record_id'] == testUser]

            event_instances = user_data[user_data[event].notna()]
            if event_instances.empty:
                continue
            for _, event_row in event_instances.iterrows():
                event_day = event_row['daysSince']

                # Define the periods for comparison
                period_before = user_data[(user_data['daysSince'] >= event_day - 16) & (user_data['daysSince'] < event_day - 4)]
                period_around = user_data[(user_data['daysSince'] >= event_day - 4) & (user_data['daysSince'] <= event_day + 4)]

                if not period_before.empty and not period_around.empty:
                    if len(period_before)<7 or len(period_around)<6:
                        continue
                    # Select a relevant column (for example, 'deep' sleep) to analyze
                    deep_before = period_before[feat_name].dropna()
                    deep_around = period_around[feat_name].dropna()

                    # Ensure both periods have enough data for t-test
                    if len(deep_before) > 1 and len(deep_around) > 1:
                        # Perform paired t-test
                        t_stat, p_val = ttest_ind(deep_before, deep_around, equal_var=False)

                        # Store the results in the DataFrame
                        if event == 'AE':
                            event_name = event_row['AE']
                        ttest_results.append({
                            'record_id': testUser,
                            'daysSince': event_day,
                            'event_type': event,
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'event_name': event_name
                        })

    ttest_results_df = pd.concat([pd.DataFrame([result]) for result in ttest_results], ignore_index=True)
    return ttest_results_df

def calculate_t_stat_range(event_impact_df):
    # Group by event type and calculate the min, max, and range of t-statistics
    t_stat_summary = event_impact_df.groupby('event_type')['mean_t_statistic'].agg(['min', 'max'])
    t_stat_summary['range'] = t_stat_summary['max'] - t_stat_summary['min']
    return t_stat_summary