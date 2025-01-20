# Find days before and after delivery as a new column
def findDaysSince(df):
    
    df2 = []
   
    # for every unique user
    for record_id in df.record_id.unique():
        
        if (len(df_birth.loc[df_birth.record_id == record_id]) != 0):
            birth = df_birth.loc[df_birth.record_id == record_id].reset_index()
            birthDate = birth['date'][0]
            # If the birth date is not null
            if (pd.isnull(birthDate) == False):
                # Get individual dataframe
                userDF = df[df['record_id'] == record_id]
                # For every date in the dataframe, add a new column value for days since birth
                userDF['daysSince'] = userDF.apply(lambda row: row.date - birthDate, axis=1)
                df2.append(userDF)
    return pd.concat(df2, ignore_index=True)


# Updated findDaysSince to accomodate the adverse event report date
def findDaysSinceAE_fixed(df):
    
    df2 = []
   
    # for every unique user
    for record_id in df.record_id.unique():
        
        if (len(df_birth.loc[df_birth.record_id == record_id]) != 0):
            birth = df_birth.loc[df_birth.record_id == record_id].copy().reset_index()
            birthDate = pd.to_datetime(birth['date'][0])  # Ensure birthDate is datetime
            # If the birth date is not null
            if (pd.isnull(birthDate) == False):
                # Get individual dataframe
                userDF = df[df['record_id'] == record_id].copy()
                # Ensure 'incident_date' is a datetime
                userDF['incident_date'] = pd.to_datetime(userDF['incident_date'])
                # For every date in the dataframe, add a new column value for days since birth
                userDF['daysSince'] = userDF.apply(lambda row: row.incident_date - birthDate, axis=1)
                df2.append(userDF)
    return pd.concat(df2, ignore_index=True)

# Compile Adverse Events and High Daily scores into one dataframe
def user_events(user, threshold=6):
    events = pd.DataFrame()
    # Daily Symptom Survey
    for severe in ['cold_infection','fever_cough','severe_headaches','vision_changes','frequent_urination']:
        part_sever = sever_sym_df[sever_sym_df['record_id'] == user].copy()
        part_sever = part_sever[part_sever[severe].notna()]
        
        for index, row in part_sever.iterrows():
            events = pd.concat([events, pd.DataFrame({'record_id': [user], 'date': [row['event_date']], 'value': [row[severe]], 'type': [severe]})], ignore_index=True)
        
    # Adverse Event Report
    userAE = testAE[testAE['record_id'] == user]
    for index, row in userAE.iterrows():
        events = pd.concat([events, pd.DataFrame({'record_id': [user], 'date': [row['date']], 'value': [row['label']], 'type': ['AE']})], ignore_index=True)
    # SAM
    userStress = stressSAM[stressSAM['record_id'] == user]
    for index, row in userStress.iterrows():
        events = pd.concat([events, pd.DataFrame({'record_id': [user], 'date': [row['date']], 'value': [row['position']], 'type': ['SAM Stress']})], ignore_index=True)
    userMood = moodSAM[moodSAM['record_id'] == user]
    for index, row in userMood.iterrows():
        events = pd.concat([events, pd.DataFrame({'record_id': [user], 'date': [row['date']], 'value': [row['position']], 'type': ['SAM Mood']})], ignore_index=True)
    userCognition = cognitionSAM[cognitionSAM['record_id'] == user]
    for index, row in userCognition.iterrows():
        events = pd.concat([events, pd.DataFrame({'record_id': [user], 'date': [row['date']], 'value': [row['position']], 'type': ['SAM Cognition']})], ignore_index=True)
    userEnergy = energySAM[energySAM['record_id'] == user]
    for index, row in userEnergy.iterrows():
        events = pd.concat([events, pd.DataFrame({'record_id': [user], 'date': [row['date']], 'value': [row['position']], 'type': ['SAM Energy']})], ignore_index=True)
    
    return events

def plotEvents(user, events, feat_name='rmssd', win_smooth=7):
    userSleep = df_sleep[df_sleep['record_id'] == user].copy()
    userSleep = findDaysSince(userSleep)
    userSleep['daysSince'] = userSleep['daysSince'] / np.timedelta64(1, 'D')
    userSleep = userSleep.sort_values(by='daysSince')
    userSleep[feat_name] = userSleep[feat_name].rolling(window=win_smooth, min_periods=1).mean()
    
    events = findDaysSince(events)
    events['daysSince'] = events['daysSince'] / np.timedelta64(1, 'D')

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rcParams.update({'figure.max_open_warning': 0})
    sns.set_theme(style='darkgrid')
    palette = sns.color_palette('muted')

    event_types = np.unique(events['type'])  # List of unique event types
    colors = plt.cm.get_cmap('tab20', len(event_types))  # Generate distinct colors
    line_styles = ['-', '--', '-.', ':']  # Different line styles

    color_mapping = {event_type: colors(i) for i, event_type in enumerate(event_types)}
    style_mapping = {event_type: line_styles[i % len(line_styles)] for i, event_type in enumerate(event_types)}



    for index, row in events.iterrows():
        if (row['daysSince'] < userSleep['daysSince'].min()) or (row['daysSince'] > userSleep['daysSince'].max()):
            continue
        ax.axvline(x=row['daysSince'], color=color_mapping[row['type']], linestyle=style_mapping[row['type']], linewidth=2, alpha=0.8)
    sns.lineplot(data=userSleep, x='daysSince', y=feat_name, color=palette[0], linewidth=3, ax=ax)

    ax.set_title(f'User {user} {feat_name} and Events')
    ax.set_xlabel('Days Before and After Delivery')
    ax.set_ylabel(f'{feat_name}')
    plt.show()
    
    # Create a separate figure for the legend
    plt.figure(figsize=(6, 4))
    for event_type, color in color_mapping.items():
        plt.plot([], [], label=event_type, color=color, linestyle=style_mapping[event_type])  # Create an empty plot for the legend

    plt.legend(title='Event Types', loc='center')
    plt.axis('off')  # Hide axes
    plt.show()

    
def impute_with_previous_mean(df_input, column_name):
    df = df_input.copy()
    min_value = df[column_name].min()  # Calculate the minimum value of the column
    for i in range(len(df)):
        if pd.isna(df[column_name].iloc[i]):  # Check if the value is missing
            if i >= 2 and not pd.isna(df[column_name].iloc[i - 1]) and not pd.isna(df[column_name].iloc[i - 2]):
                df[column_name].iloc[i] = (df[column_name].iloc[i - 1] + df[column_name].iloc[i - 2]) / 2
            else:
                df[column_name].iloc[i] = min_value
    return df[column_name]