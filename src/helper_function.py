import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import ast
import csv
import io
from io import StringIO, BytesIO, TextIOWrapper
import gzip
from datetime import datetime, date
import seaborn as sns
import datetime as dt
import scipy
from datetime import timedelta

def findDaysSince(df, record_ids, df_birth):
    df2 = []
    # for every unique user
    for record_id in record_ids:
        birth = df_birth.loc[df_birth.record_id == record_id, 'date']
        birthDate = birth.iloc[0]
        # If the birth date is not null
        if ((birthDate) != np.nan):
            userDF = df.loc[df.record_id == record_id].copy()
            if(len(userDF)==0):
                continue
            userDF['daysSince'] = (userDF['date']-birthDate) / np.timedelta64(1, 'D')
            df2.append(userDF)
    return pd.concat(df2, ignore_index=True)

def id_missing_info(dfs, ids_list, trimester=True, drop_dup=False):
    #bbd: before birth_date, bd: birth_date, abd: after birth_date
    #m, d, and per stands for months, days, and percentage, respectively.
    df_missing_info = pd.DataFrame(columns=['record_id','start_bbd','end_bbd','len_d_1st_tri','len_d_2nd_tri','len_d_3rd_tri','miss_per_m', 'monthly_miss_bbd', 'miss_bbd_per','bd','dur_abd','miss_abd_per'])
    for index, record_id in enumerate(ids_list):
        id_list = []
        pdf = dfs.loc[dfs.record_id == record_id].copy()
        pdf = pdf.loc[:, ['date','daysSince','record_id']]
        pdf = pdf.sort_values(by='date', ascending = False)
        if(drop_dup):
            pdf.drop_duplicates(subset='daysSince', inplace=True)
        id_list.append(record_id) #record_id
        pdf_bbd = pdf.loc[pdf.daysSince<0].copy()
        
        if(len(pdf_bbd)<=1):
            for _ in range(len(df_missing_info.columns)-4):
                id_list.append(0)
        else:
            start_time = pdf_bbd['daysSince'].iloc[-1]
            id_list.append(round(-1 * start_time/ 30, 2)) #start_bbd in m unit
            end_time = pdf_bbd['daysSince'].iloc[0]
            id_list.append(round(-1 * end_time/ 30, 2)) #end_bbd in m unit
            miss_m = []
            len_d_trimesters = np.zeros(3)
            if(trimester):
                day_dur = [-270, -180, -90, 0]
            else:
                day_dur = [-90, -60, -30, 0]
            for i, month in enumerate(np.arange(start_time, end_time, 30)):
                if(i == int((-start_time + end_time)/30)): month_end = end_time + 1;
                else: month_end = month + 30
                pdf_part = pdf_bbd.loc[(pdf_bbd.daysSince>=(month)) & (pdf_bbd.daysSince<month_end)]
                miss_m.append(pdf_part['daysSince'].diff(-1).sum() - len(pdf_part) + 1)
                if((month>=day_dur[0]) & (month<day_dur[1])): len_d_trimesters[0] += len(pdf_part)
                if((month>=day_dur[1]) & (month<day_dur[2])): len_d_trimesters[1] += len(pdf_part)
                if((month>=day_dur[2]) & (month<day_dur[3])): len_d_trimesters[2] += len(pdf_part)
            for t in range(len(len_d_trimesters)):
                id_list.append(len_d_trimesters[t]) #len of data in each trimester
            id_list.append(round(np.mean(miss_m),3)) #miss_per_m in d unit
            id_list.append(round(100 * np.sum(miss_m)/( - start_time + end_time), 3)) #sum of the data missing during recording in each month divided by the recording length
            m_bbd = pdf_bbd['daysSince'].diff(-1).sum() - len(pdf_bbd) + 1
            id_list.append(round(100 * m_bbd/( - start_time + end_time), 3)) #miss_bbd_per
        
        pdf_bd = pdf.loc[pdf.daysSince==0]
        if(len(pdf_bd)==1): id_list.append(1);
        else: id_list.append(0) #if bd information exists 
        
        pdf_abd = pdf.loc[pdf.daysSince>0]
        if(len(pdf_abd)<=1): 
            id_list.append(0); id_list.append(0)
        else:
            id_list.append(round(len(pdf_abd)/30,3)) #dur_abd in m unit
            m_abd = pdf_abd['daysSince'].diff(-1).sum() - len(pdf_abd) + 1
            id_list.append(round(100 * m_abd/(-pdf_abd['daysSince'].iloc[-1] + pdf_abd['daysSince'].iloc[0]), 3)) #miss_abd_per
        
        #print(index, id_list)
        df_missing_info.loc[index] = id_list
        
    return df_missing_info

def id_missing_bump_group(dfs, ids_list, trimester=True, drop_dup=False):
    #bbd: before birth_date, bd: birth_date, abd: after birth_date
    #m, d, and per stands for months, days, and percentage, respectively.
    df_missing_info = pd.DataFrame(columns=['record_id','start_bbd','end_bbd','len_d_1st_tri','len_d_2nd_tri','len_d_3rd_tri','miss_per_m', 'monthly_miss_bbd', 'miss_bbd_per','dur_abd','miss_abd_per','num_day', 'miss_dur_per'])
    for index, record_id in enumerate(ids_list):
        id_list = []
        pdf = dfs.loc[dfs.record_id == record_id].copy()
        pdf = pdf.loc[:, ['date','daysSince','record_id']]
        pdf = pdf.sort_values(by='date', ascending = False)
        if(drop_dup):
            pdf.drop_duplicates(subset='daysSince', inplace=True)
        id_list.append(record_id) #record_id
        pdf_bbd = pdf.loc[pdf.daysSince<0].copy()
        
        if(len(pdf_bbd)<=1):
            for _ in range(len(df_missing_info.columns)-4):
                id_list.append(0)
        else:
            start_time = pdf_bbd['daysSince'].iloc[-1]
            id_list.append(round(-1 * start_time/ 30, 2)) #start_bbd in m unit
            end_time = pdf_bbd['daysSince'].iloc[0]
            id_list.append(round(-1 * end_time/ 30, 2)) #end_bbd in m unit
            miss_m = []
            len_d_trimesters = np.zeros(3)
            if(trimester):
                day_dur = [-270, -180, -90, 0]
            else:
                day_dur = [-90, -60, -30, 0]
            for i, month in enumerate(np.arange(start_time, end_time, 30)):
                if(i == int((-start_time + end_time)/30)): month_end = end_time + 1;
                else: month_end = month + 30
                pdf_part = pdf_bbd.loc[(pdf_bbd.daysSince>=(month)) & (pdf_bbd.daysSince<month_end)]
                miss_m.append(pdf_part['daysSince'].diff(-1).sum() - len(pdf_part) + 1)
                if((month>=day_dur[0]) & (month<day_dur[1])): len_d_trimesters[0] += len(pdf_part)
                if((month>=day_dur[1]) & (month<day_dur[2])): len_d_trimesters[1] += len(pdf_part)
                if((month>=day_dur[2]) & (month<day_dur[3])): len_d_trimesters[2] += len(pdf_part)
            for t in range(len(len_d_trimesters)):
                id_list.append(len_d_trimesters[t]) #len of data in each trimester
            id_list.append(round(np.mean(miss_m),3)) #miss_per_m in d unit
            id_list.append(round(100 * np.sum(miss_m)/( - start_time + end_time), 3)) #sum of the data missing during recording in each month divided by the recording length
            m_bbd = pdf_bbd['daysSince'].diff(-1).sum() - len(pdf_bbd) + 1
            id_list.append(round(100 * m_bbd/( - start_time + end_time), 3)) #miss_bbd_per
        
        
        pdf_abd = pdf.loc[pdf.daysSince>0]
        if(len(pdf_abd)<=1): 
            id_list.append(0); id_list.append(0)
        else:
            id_list.append(round(len(pdf_abd)/30,3)) #dur_abd in m unit
            m_abd = pdf_abd['daysSince'].diff(-1).sum() - len(pdf_abd) + 1
            id_list.append(round(100 * m_abd/(-pdf_abd['daysSince'].iloc[-1] + pdf_abd['daysSince'].iloc[0]), 3)) #miss_abd_per

        pdf_dur = pdf.loc[(pdf.daysSince<100)&(pdf.daysSince>-200)].copy()
        if(len(pdf_dur)<=1): 
            id_list.append(0); id_list.append(0)
        else:
            if((pdf_bbd['daysSince'].iloc[-1]<=-200)&(pdf_bbd['daysSince'].iloc[0]>=100)):
                print(pdf_bbd['daysSince'].iloc[-1], pdf_bbd['daysSince'].iloc[0], pdf_bbd.record_id)
            m_abd = pdf_dur['daysSince'].diff(-1).sum() - len(pdf_dur) + 1
            id_list.append(-pdf_dur['daysSince'].iloc[-1] + pdf_dur['daysSince'].iloc[0]) #num_day
            id_list.append(round(100 * m_abd/(-pdf_dur['daysSince'].iloc[-1] + pdf_dur['daysSince'].iloc[0]), 3)) #miss_dur_per

        #print(index, id_list)
        df_missing_info.loc[index] = id_list
        
    return df_missing_info

def print_missing_behaviour(dfs, unique_ids=None, record_id=None, abd=False, plot_flag=False, only_bbd=True):
    #df_missing_info = pd.DataFrame(columns=['user_id','daysSince','missing_map'])
    if(record_id != None):
        unique_ids = [record_id]
    df2 = []
    for index, rec_id in enumerate(unique_ids):
        id_list = []
        pdf = dfs.loc[dfs.record_id == rec_id,['record_id','daysSince']].copy()
        pdf.sort_values(by='daysSince', ascending = True, inplace=True)
        if(only_bbd):
            pdf = pdf.loc[pdf.daysSince<=0]
        pdf.index = pdf.daysSince
        if(len(pdf) < 1):
            continue
        idx = np.arange(pdf.daysSince.iloc[0],pdf.daysSince.iloc[-1]+1,1)
        #print(pdf.daysSince.iloc[0],pdf.daysSince.iloc[-1], idx[0], idx[-1])
        dup_row = pdf[pdf.index.duplicated()]
        if(len(dup_row)>0):
            print('duplicate row: ', dup_row)
            pdf = pdf.drop_duplicates(subset=['daysSince'])
        pdf = pdf.reindex(idx, fill_value=np.nan)
        pdf.record_id.fillna(rec_id, inplace=True)
        miss_flag = list(pdf.daysSince.isna().astype('int').values)
        pdf.daysSince = list(idx)
        pdf['miss_flag'] = miss_flag
        #num_miss = (pdf.miss_flag.sum())
        pdf['seq_miss'] = (pdf.miss_flag.diff(1) != 0).astype('int').cumsum()
        #u = (pdf.groupby(by = 'seq_miss').apply(lambda x: (len(x)))).max()
        #print(f'user id: {record_id} has {num_miss} missing days')
        if(plot_flag):
            if(num_miss != 0):
                pdf.plot(y='miss_flag', style='.-', title='missing pattern for user id: ' + str(rec_id))
        df2.append(pdf)
    return pd.concat(df2, ignore_index=True)
        
def exploreDataBirthBA(df, col, record_id):
    """
    Creates a plot of a single user and variable with date of delivery and means before and after
    """
    # Plot initializations
    plt.rcParams.update({'figure.max_open_warning': 0})
    sns.set(style='darkgrid')
    plt.figure(figsize=(12,4))
    pdf = df.loc[df.record_id == record_id]
    pdf = pdf.sort_values(by='date', ascending = True)
    pdf.dropna(inplace=True)
    pdf = pdf.drop_duplicates(subset='date')
    
    g = sns.scatterplot(data=pdf, x='date', y=col, ci=None, color='purple')
    g.set(xlim=(pdf.date.iloc[0], pdf.date.iloc[-1]))

    sns.lineplot(data=pdf, x='date', y=col, ci=None, markers=True)
    # If there is a date of delivery
    if (len(df_birth.loc[df_birth.record_id == record_id]) != 0):
        birth = df_birth.loc[df_birth.record_id == record_id].reset_index()
        # Plot a vertical line the length of the graph
        plt.axvline(x=birth.date, color = 'y', ls='--')
        ymin, ymax = plt.gca().get_ylim()
        xmin, xmax = plt.gca().get_xlim()
        plt.text(birth.date, ymax, birth['date'][0], fontsize=12, color='y')
        
        # Get means before and after delivery
        
        #print(pdf.date.iloc[[0, -1]], birth.date)
        xmin, xmax = pdf.date.iloc[0], pdf.date.iloc[-1]
        after = pdf[~(pdf['date'] < birth.date[0])]
        before = pdf[~(pdf['date'] > birth.date[0])]
        before_avg = before[col].mean()
        after_avg = after[col].mean()

        # Plot means 
        plt.hlines(y=before_avg, xmin=xmin, xmax=birth.date, color='blue', linestyles='dashdot')
        plt.hlines(y=after_avg, xmin=birth.date, xmax=xmax, color='red', linestyles='dashdot')

    plt.xlabel(''); plt.ylabel(col)
    plt.title('Record ID: ' + str(record_id))
    plt.show
    
def match_dfs(df, ids):
    joint_id = [x for x in df.record_id.unique() if x in ids]
    print(f'number of joint id: {len(joint_id)} and number of all ids: {len(df.record_id.unique())}, {len(ids)}')
    df2 = df.loc[df.record_id.isin(joint_id)].copy()
    return df2

def label_dist(df_label, df_rec=None):
    ids_list = df_label.record_id.unique()
    q_idx_list = df_label.question_id.unique()
    df_label_info = pd.DataFrame(columns=['record_id','n_miss_inner'])
    if len(df_rec) != 0:
        df_label_info['n_miss_outer_b'] = None
        df_label_info['n_miss_outer_a'] = None

    for q_id in q_idx_list:
        df_label_info[str(q_id)] = None
    for index, idx in enumerate(ids_list):
        id_list = []
        pdf = df_label.loc[df_label.record_id==idx].copy()
        pdf = pdf.loc[pdf.daysSince<0]
        if(len(pdf) == 0):
            continue
        if(len(pdf) == 1):
            print(idx, 'with length of 1')
            continue
        id_list.append(idx)
        pdf = pdf.loc[:, ['date','daysSince','record_id', 'question_id', 'answer_text']]
        pdf = pdf.sort_values(by='date', ascending = False)
        df = pdf.loc[pdf.question_id==213]
        id_list.append((- df.daysSince.iloc[-1] + df.daysSince.iloc[0])/14 + 1 - len(df))
            
        if len(df_rec) != 0:
            pdf_rec = df_rec.loc[df_rec.record_id==idx].copy()
            pdf_rec = pdf_rec.loc[:, ['date','daysSince']]
            pdf_rec = pdf_rec.sort_values(by='date', ascending = False)
            pdf_rec = pdf_rec.loc[pdf_rec.daysSince<0]
            #print('here:', (df.daysSince.iloc[-1] - pdf_rec.daysSince.iloc[-1])//14, 'and', (- df.daysSince.iloc[0] + pdf_rec.daysSince.iloc[0])//14)

            miss_outer_b = ((df.daysSince.iloc[-1] - pdf_rec.daysSince.iloc[-1])//14) 
            miss_outer_a = ((- df.daysSince.iloc[0] + pdf_rec.daysSince.iloc[0])//14)
            id_list.append((float(miss_outer_b)))
            id_list.append((float(miss_outer_a)))

        for q_id in q_idx_list:
            id_list.append(float(len(pdf.loc[pdf.question_id==q_id].loc[pdf.answer_text=='Yes'])))
            
        df_label_info.loc[index] = id_list
    return df_label_info

def severe_label(df_label, df_rec=None):
    ids_list = df_label.record_id.unique()
    q_idx_list = df_label.question_id.unique()
    df_label_new = pd.DataFrame(columns=['record_id','daysSince'])
    for q_id in q_idx_list:
        df_label_new[str(q_id)] = None
        
    i = 0
    for index, idx in enumerate(ids_list):
        id_list = []
        pdf = df_label.loc[df_label.record_id==idx].copy()
        pdf = pdf.loc[pdf.daysSince<0]
        if(len(pdf) == 0):
            continue
        if(len(pdf) == 1):
            print(idx, 'with length of 1')
            continue
        pdf = pdf.loc[:, ['date','daysSince','record_id', 'question_id', 'answer_text']]
        pdf = pdf.sort_values(by='date', ascending = False)
        
        for j, day in enumerate(pdf.daysSince.unique()):
            id_list = []
            id_list.append(idx)
            id_list.append(day)
            for q_id in q_idx_list:
                mid = pdf.loc[(pdf.daysSince==day) & (pdf.question_id==q_id)]
                if(len(mid) == 0):
                    id_list.append(np.nan)
                elif(list(mid.answer_text=='Yes')[0]):
                    id_list.append(1)
                else:
                    id_list.append(0)
            df_label_new.loc[i] = id_list
            i += 1
            
    return df_label_new

def features_corr(df, name=''):
    corr_m = abs(df.corr())
    print('correlation shape: ', corr_m.shape)
    plt.matshow(corr_m)
    plt.xticks(range(corr_m.select_dtypes(['number']).shape[1]), corr_m.select_dtypes(['number']).columns, fontsize=8, rotation=75.)
    plt.yticks(range(corr_m.shape[1]), corr_m.columns, fontsize=8)
    plt.tick_params(axis='x',labelbottom=True, labeltop=False, top=False)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.gcf().set_size_inches(6, 6)
    plt.title(name + 'Correlation Matrix', fontsize=16);
    plt.show()
    
def rm_outliner_pbyp(in_df):
    df2 = []
    for idx in in_df.record_id.unique():
        dfs = in_df.loc[in_df.record_id==idx].copy()
        dfs = dfs.sort_values(by='daysSince', ascending = False)
        for feat in list(dfs.columns)[:-4]:
            
            q_low = dfs.loc[:,feat].quantile(0.05)
            q_high = dfs.loc[:,feat].quantile(0.09)
            dfs.loc[((dfs.loc[:,feat] > q_high) | (dfs.loc[:,feat] < q_low)), feat] = np.nan
            
        df2.append(dfs)
    df2 = pd.concat(df2, ignore_index=True)
    return df2

def rm_outliner(in_df, compare_ids=None, feat_list=None):
    df2 = in_df.copy()
    q_low = {}
    q_high = {}
    if(compare_ids == None):
        compare_ids = list(in_df.record_id.unique())
    if(feat_list == None):
        feat_list = list(dfs.columns)[:-4]
        #dfs = in_df.loc[in_df.record_id.apply(lambda x: x == idx[:10])]
    dfs = [in_df[in_df['record_id'] == id] for id in compare_ids[:100]]
    dfs = pd.concat(dfs, ignore_index=True)
    #print(dfs.head())
    for feat in feat_list:
        q_low[feat] = dfs.loc[:,feat].quantile(0.05)
        q_high[feat] = dfs.loc[:,feat].quantile(0.95)
            
    for feat in feat_list:
        df2.loc[((df2.loc[:,feat] > q_high[feat]) | (dfs.loc[:,feat] < q_low[feat])), feat] = np.nan
    return df2, q_low, q_high

def df_impute(df_features):
    df2 = []
    for idx in df_features.record_id.unique():
        dfs = df_features.loc[df_features.record_id==idx].copy()
        if(dfs.isna().sum().max() == len(dfs)):
            print(f'for user {idx} the full length of feature is nan')
            #print(dfs.isna().sum())
            continue
        dfs = dfs.sort_values(by='daysSince', ascending = False)
        dfs.iloc[:,:-4].interpolate(inplace=True, method='linear', axis=0, limit=5)
        #df_final = dfs.fillna(1)
        num_nan = sum(dfs.isna().sum())
        if(num_nan >=1):
            
            dfs.bfill(inplace=True, axis=0)
            dfs.ffill(inplace=True, axis=0)
            if(sum(dfs.isna().sum()) != 0):
                print(dfs.loc[dfs.isna().any(axis=1)]) 
            #print(sum(df_final.isna().sum()), df_final.loc[df_final.isna().any(axis=1)])
            #print(dfs.iloc[:3])
        df2.append(dfs)
    df2 = pd.concat(df2, ignore_index=True)
    return df2
    
def find_bad_record_id(df_missing, df2, record_ids):
    miss_sum = df_missing.groupby(by='record_id').apply(lambda x: (x.miss_flag==1).sum()/len(x))
    id_bad_users = miss_sum.loc[miss_sum>.33].index.values
    df_missing_info = id_missing_info(df2, record_ids)
    id_short = df_missing_info.loc[((df_missing_info['dur_bbd']<=3) & (df_missing_info['dur_bbd']>=0)), 'record_id'].values
    record_ids_clean = [name for name in record_ids if not((name in id_bad_users) or (name in id_short))]
    return record_ids_clean
    
def all_in_one(df, record_ids, df_birth, record_ids_clean=None, with_outliner=True, feat_list=None):
    df2 = findDaysSince(df, record_ids, df_birth)
    if(with_outliner):
        df2, _, _ = rm_outliner(df2, feat_list=feat_list)
    df_missing = print_missing_behaviour(df2, unique_ids = record_ids, only_bbd=True)
    df3 = df2.loc[df2.record_id.isin(record_ids_clean)]
    df3 = df3.loc[df3.daysSince <=0]
    df3 = df3.drop_duplicates(subset=['daysSince','record_id'])
    
    if(record_ids_clean == None):
        record_ids_clean = find_bad_record_id(df_missing, df2, record_ids)
        
    df2_missing = df_missing.loc[df_missing.record_id.isin(record_ids_clean)]
    df3_merged = df3.merge(df2_missing, how='right', on=['record_id','daysSince'])
    print('merged dataframes shape: ', df3_merged.shape)
    return df3_merged

def df_norm(df, method='initial_w', feat_cols=None, minmax=True):
    df2 = []
    if(feat_cols == None):
        feat_cols = list(df.columns)        
    if('miss_flag' in feat_cols):
        feat_cols.remove('miss_flag')
    for i, x in enumerate(df.record_id.unique()):
        df_i = df.loc[df.record_id == x]
        df_i = df_i.sort_values(by='daysSince', ascending = False)
        if(method=='initial_w'):
            if(minmax):
                df_min = df_i.loc[:,feat_cols].iloc[:14].min()
                df_max = df_i.loc[:,feat_cols].iloc[:14].max()
                df_max[df_max<1] = 1
                df_i.loc[:,feat_cols] = df_i.loc[:,feat_cols] / (df_max + np.finfo(np.float16).eps)
                #df_i[feat_cols] = (df_i[feat_cols] - df_min) / (df_max - df_min + np.finfo(np.float16).eps)
            else:
                df_i[feat_cols] = df_i[feat_cols].apply(lambda x: (x - x.iloc[:14].mean()) / (x.iloc[:14].std() + np.finfo(np.float16).eps))
        elif(method=='features'):
            if(minmax):
                df_i[feat_cols] = df_i[feat_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min() + np.finfo(np.float16).eps), axis=1)
            else:
                df_i[feat_cols] = df_i[feat_cols].apply(lambda x: (x - x.mean()) / (x.std()+ np.finfo(np.float16).eps), axis=1)
        else:
            assert('wrong normalization format')
        df2.append(df_i.copy())
    return pd.concat(df2, ignore_index=True)

