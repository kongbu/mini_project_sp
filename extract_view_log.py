import pandas as pd
import numpy as np

def voucher_day_feature(): #merge by userid and voucher date
    view_log_0 = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/view_log_0.csv')
    view_log_0.isnull().sum() #571381 no activate in event_name
    view_log_0[['event_name']] = view_log_0[['event_name']].fillna(value='no_action')
    pivot_df =  pd.pivot_table(view_log_0, values='count', index=['userid','voucher_received_date'], columns='event_name', aggfunc=sum).reset_index() 
    pivot_df = pivot_df.fillna(value=0)
    pivot_df.to_csv('voucher_day_actions.csv', index=False)


def extract_view_log_feature(): #historical feature for user and voucher date pair
    view_df_30days = []
    for i in range(30):
        filename = 'D:/Projects/ShopeeMiniProject/Data/view_log_' + str(i+1) + '.csv'
        view_df = pd.read_csv(filename)
        view_df[['event_name']] = view_df[['event_name']].fillna(value='no_action')
        #add i+1 day
        view_df['voucher_received_date'] = pd.DatetimeIndex(view_df.iloc[:,1]) + pd.DateOffset(i+1)
        view_df['voucher_received_date'] = view_df['voucher_received_date'].apply(lambda x: str(x)[:10])
        #drop the seond column
        view_df = view_df.drop(view_df.columns[1], axis=1)
        view_df_30days.append(view_df)
    view_df_30days = pd.concat(view_df_30days, axis=0)
    mean_30days =  pd.pivot_table(view_df_30days, values='count', index=['userid','voucher_received_date'], columns='event_name', aggfunc=mean).reset_index() 
    mean_30days.columns = np.where(mean_30days.columns.isin(mean_30days.columns[2:]), mean_30days.columns+'_pre30days_mean', mean_30days.columns)
    mean_30days = mean_30days.fillna(value=0)
    sum_30days =  pd.pivot_table(view_df_30days, values='count', index=['userid','voucher_received_date'], columns='event_name', aggfunc=sum).reset_index() 
    sum_30days.columns = np.where(sum_30days.columns.isin(sum_30days.columns[2:]), sum_30days.columns+'_pre30days_sum', sum_30days.columns)
    sum_30days = sum_30days.fillna(value=0)
    median_30days =  pd.pivot_table(view_df_30days, values='count', index=['userid','voucher_received_date'], columns='event_name', aggfunc=median).reset_index() 
    median_30days.columns = np.where(median_30days.columns.isin(median_30days.columns[2:]), median_30days.columns+'_pre30days_median', median_30days.columns)
    median_30days = median_30days.fillna(value=0)
    min_30days =  pd.pivot_table(view_df_30days, values='count', index=['userid','voucher_received_date'], columns='event_name', aggfunc=min).reset_index() 
    min_30days.columns = np.where(min_30days.columns.isin(min_30days.columns[2:]), min_30days.columns+'_pre30days_min', min_30days.columns)
    min_30days = min_30days.fillna(value=0)
    max_30days =  pd.pivot_table(view_df_30days, values='count', index=['userid','voucher_received_date'], columns='event_name', aggfunc=max).reset_index() 
    max_30days.columns = np.where(max_30days.columns.isin(max_30days.columns[2:]), max_30days.columns+'_pre30days_max', max_30days.columns)
    max_30days = max_30days.fillna(value=0)
    std_30days =  pd.pivot_table(view_df_30days, values='count', index=['userid','voucher_received_date'], columns='event_name', aggfunc=std).reset_index() 
    std_30days.columns = np.where(std_30days.columns.isin(std_30days.columns[2:]), std_30days.columns+'_pre30days_std', std_30days.columns)
    std_30days = std_30days.fillna(value=0)
    # merge them
    merged_data = mean_30days.merge(sum_30days,on=['userid','voucher_received_date']).merge(median_30days,on=['userid','voucher_received_date']).merge(min_30days,on=['userid','voucher_received_date']).merge(max_30days,on=['userid','voucher_received_date']).merge(std_30days,on=['userid','voucher_received_date'])
    merged_data.to_csv('view_pre30days_feature.csv', index=False)





if __name__ == "__main__":
    extract_view_log_feature()