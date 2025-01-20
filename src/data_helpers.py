import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
    
def calculate_bmi(weight_lbs, height_feet, height_inches):
    #In metric units: BMI = weight (kg) ÷ height2 (meters)
    #In US units: BMI = weight (lb) ÷ height2 (inches) * 703

    # Convert height to inches
    total_height_inches = height_feet * 12 + height_inches
    bmi = (weight_lbs / (total_height_inches ** 2)) * 703
    return bmi

def load_static_feat(file_loc, ids_list, df_birth, impute=True, parity=True, pred_cond=True):
    #Extract static demographic features of participants
    key = 'app_activities/surveys.csv.gz'
    df_survey = pd.read_csv(file_loc+key, compression='gzip') #dtype={"user_id": int, "username": "string"}
    df_survey_new = df_survey.loc[df_survey.record_id.isin(ids_list)]
    
    if parity:
        n_prev_baby = df_survey_new.loc[:,['record_id','answer_text']].loc[df_survey_new.question_id==264]
        n_prev_baby.drop_duplicates(inplace=True)
        n_prev_baby = n_prev_baby.iloc[:-2]
        n_prev_baby['n_prev_baby'] = pd.to_numeric(n_prev_baby['answer_text'])
        n_prev_baby.drop(columns=['answer_text'], inplace=True)

    age = df_survey_new.loc[:,['record_id','question_text','answer_text']].loc[df_survey_new.question_id==167]
    age['answer_text'] = pd.to_datetime(age.answer_text).dt.date
    age.drop_duplicates(inplace=True)
    age = age.groupby("record_id").agg(min_date=('answer_text', 'min'))
    df_birth2 = df_birth.drop_duplicates(subset=['record_id']) 
    age = age.merge(df_birth2.loc[:,['birth_date','record_id','birth_gestage']], how='left', on='record_id')
    age['birth_date'] = pd.to_datetime(age.birth_date).dt.date
    age['gestage_week'] = age['birth_gestage'].str.split("w|,| ", n=1).str[0]
    age['gestage_week'] = age['gestage_week'].str.replace(">", '')
    #age['gestage_week'].loc[age['gestage_week'].str.contains('\\+', na=False)] = np.nan
    age['gestage_week'] = age['gestage_week'].str.replace("+", '')
    age = age.dropna(subset=['birth_gestage'])
    age['gestage_week'] = pd.to_numeric(age['gestage_week'])
    age['age'] = (age.birth_date-age.min_date)/ np.timedelta64(1, 'W')#.astype(int)
    age['age'] = (age['age'] - age['gestage_week'])//(4*12)
    age.drop(columns=['birth_gestage','birth_date','min_date'], inplace=True)
    if pred_cond:
        prev_condition_q = df_survey_new.question_text.loc[df_survey.question_text.str.contains('condition')].unique()
        for i in range(len(prev_condition_q)):
            df = df_survey_new.loc[:,['record_id','answer_text', 'question_id','id']].loc[df_survey_new.question_text==(prev_condition_q[i])].copy()
            df.drop_duplicates(inplace=True)
            df = df.loc[df.groupby('record_id').id.idxmin()].reset_index(drop=True)
            df.rename(columns={"answer_text": str(df.question_id.iloc[1])},inplace=True)
            df.drop(['question_id','id'], axis=1, inplace=True)
            if(i==0):
                prev_cond = df
            else:
                prev_cond = prev_cond.merge(df,how='right',on='record_id')
        prev_cond = prev_cond.replace(['No','Yes','Not sure'],[0,1,0.5])
        prev_cond = prev_cond.drop(columns=['233', '235', '237','238','247','249', '258', '259']) #Heart Disease: 3, Kidney Disease: 3

    key = 'redcap/personal_characteristics.csv.gz'
    df_personal = pd.read_csv(file_loc+key, compression='gzip')
    df_personal_new = df_personal.loc[df_personal.record_id.isin(ids_list)].copy()
    df_weight = df_personal_new.loc[:,['weight_prepreg','weight','record_id']].drop_duplicates()
    lin_model = LinearRegression()
    loc_id = (df_weight['weight']=='134  (124 before ivf)')
    df_weight.loc[loc_id, ['weight_prepreg', 'weight']] = ['124', '134']
    df_weight.weight.loc[df_weight.weight=='UNKNOWN'] = np.nan
    df_weight['weight'] = pd.to_numeric(df_weight['weight'])
    df_weight['weight_pre'] = df_weight.weight_prepreg.str.split("-| ", n=1).str[0]
    df_weight['weight_pre'] = pd.to_numeric(df_weight['weight_pre'])
    df_weight = df_weight.sort_values(by='weight')
    df2 = df_weight.dropna().copy()
    x = np.array(df2.weight.values).reshape(-1,1)
    y = np.array(df2.weight_pre.values).reshape(-1)
    lin_model.fit(x, y)
    df_weight = df_weight.loc[df_weight.weight.notna()]
    pre_loc = df_weight.weight_pre.isna()
    df_weight.weight_pre.loc[pre_loc] = list(lin_model.intercept_ + lin_model.coef_ * df_weight.weight.loc[pre_loc])
    df_weight.drop(columns=['weight_prepreg','weight'], inplace=True)
    
    df_height = df_personal_new.loc[:,['height_feet','height_inches','record_id']].copy().drop_duplicates()
    loc_id = (df_height['height_feet']=='5 feet 7 inches')
    df_height.loc[loc_id, ['height_feet', 'height_inches']] = ['5', 7]
    df_height.height_inches.fillna(0, inplace=True)
    df_height.loc[df_height.height_inches>=12, ['height_inches']] = 6
    df_height['height_feet'] = df_height.height_feet.str.split("\'", n=1).str[0]
    df_height['height_feet'] = df_height.height_feet.str.split("\.", n=1).str[0]
    df_height['height_feet'] = pd.to_numeric(df_height['height_feet'])
    df_height = df_height.merge(df_weight,how='inner',on='record_id')
    df_height['bmi'] = calculate_bmi(df_height.weight_pre.values, df_height.height_feet.values, df_height.height_inches.values)

    static_feat = age.copy()
    if pred_cond:
        static_feat = static_feat.merge(prev_cond,how='left',on='record_id')
    static_feat = static_feat.merge(df_weight,how='left',on='record_id')
    static_feat = static_feat.merge(df_height[['record_id', 'bmi']],how='left',on='record_id')
    if parity:
        static_feat = static_feat.merge(n_prev_baby,how='left',on='record_id')
    if impute:
        if parity:
            static_feat.n_prev_baby.fillna(0, inplace=True)
        static_feat.weight_pre.fillna(static_feat.weight_pre.mean(), inplace=True)
        static_feat.bmi.fillna(static_feat.bmi.mean(), inplace=True)
        #static_feat = static_feat.loc[static_feat.age.notna()]
        static_feat.fillna(0.0, inplace=True)
    return static_feat


def scale_normalize(x, y):
    return x/(np.nanmax(y)-np.nanmin(y))

def normalize(x):
    x = (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x)) # normalize x between 0 and 1
    return x
              
def normalize_between_minus_one_and_one(x):
    x = 2*(normalize(x) - 0.5)
    return x

def roll_signal(df, field, only_mean=False, twice=False, normalize=False, user_col='user_id'):
    cols = list(df.columns)
    df['rolled_' + field] = df.groupby(user_col)[field].transform(lambda x: x.rolling(8, min_periods=1).mean().shift(-4))
    df[field + '_std'] = df.groupby(user_col)[field].transform(lambda x: x.rolling(8, min_periods=1).std().shift(-4))
    if twice:
        df['rolled_' + field] = df.groupby(user_col)['rolled_' + field].transform(lambda x: x.rolling(8, min_periods=1).mean().shift(-4))
    
    if normalize:
        df['scale'] = df.groupby(user_col)['rolled_' + field].transform(lambda x: x.max() - x.min())
        df[field + '_std'] = df[field + '_std']/df.scale
        df['rolled_' + field] = df.groupby(user_col)['rolled_' + field].transform(normalize)
    
    df['rolled_' + field + '_minus'] = df['rolled_' + field] - df[field + '_std']/2
    df['rolled_' + field + '_plus'] = df['rolled_' + field] + df[field + '_std']/2
    if only_mean:
        df = df[cols + ['rolled_' + field]]
    return df