import pandas as pd
import time
import datetime as DT
import itertools
from scipy.sparse import csc_matrix
from sparsesvd import sparsesvd
import numpy as np

# merge training and testing with user frofile, voucher_distribution_active_date and Voucher_mechanics first
training_data = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/training.csv')
testing_data = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/predict.csv')
usr_profile = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/user_profiles_MY.csv')
active_sess = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/voucher_distribution_active_date.csv')
voucher_info = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/Voucher_mechanics.csv')

# merge user profile
training_data = pd.merge(training_data, usr_profile, how='left', on=['userid'])
testing_data = pd.merge(testing_data, usr_profile, how='left', on=['userid'])
training_data.isnull().sum() # 228969 32.3% nan gender, 371828 52.4% nan birthday, too many missing.
#calculate age
now = pd.Timestamp(DT.datetime.now())
training_data['dob'] = pd.to_datetime(training_data['birthday'], format='%Y-%m-%d')
testing_data['dob'] = pd.to_datetime(testing_data['birthday'], format='%Y-%m-%d')
training_data['age'] = (now - training_data['dob']).astype('<m8[Y]')
testing_data['age'] = (now - testing_data['dob']).astype('<m8[Y]')
#generate months since register.
training_data['dob2'] = pd.to_datetime(training_data['registration_time'], format='%Y-%m-%d %H:%M:%S')
testing_data['dob2'] = pd.to_datetime(testing_data['registration_time'], format='%Y-%m-%d %H:%M:%S')
training_data['months_since_register'] = (now - training_data['dob2']).astype('<m8[M]')
testing_data['months_since_register'] = (now - testing_data['dob2']).astype('<m8[M]')

# merge voucher_distribution_active_date
active_sess = active_sess.fillna(0) #fill zeros to nan of the number of active sessions
training_data = pd.merge(training_data, active_sess, how='left', on=['userid', 'promotionid_received','voucher_code_received','voucher_received_time'])
testing_data = pd.merge(testing_data, active_sess, how='left', on=['userid', 'promotionid_received','voucher_code_received','voucher_received_time'])
# add mean, median, max, std.
active_sess['mean_active_sess'] = active_sess.iloc[:, 5:-1].mean(axis=1)
active_sess['max_active_sess'] = active_sess.iloc[:, 5:-1].max(axis=1)
active_sess['median_active_sess'] = active_sess.iloc[:, 5:-1].median(axis=1)
active_sess['std_active_sess'] = active_sess.iloc[:, 5:-1].std(axis=1)
added_feas = active_sess.filter(['userid', 'promotionid_received','voucher_code_received','voucher_received_time','mean_active_sess', 'max_active_sess','median_active_sess','std_active_sess'], axis=1)

# merge Voucher_mechanics
training_data = pd.merge(training_data, voucher_info, how='left', on=['promotionid_received'])
testing_data = pd.merge(testing_data, voucher_info, how='left', on=['promotionid_received'])

training_data1 = training_data.drop(['registration_time','birthday','dob','dob2', 'voucher_received_date'],axis=1)
testing_data = testing_data.drop(['registration_time','birthday','dob','dob2'],axis=1)
training_data1.to_csv('training_data1.csv', index=False)

# extract fea in likes.csv

# add svd features
# set work directory os.chdir('D:/pythoncode/shopeeMiniProject/shopeeMiniProject') get os.getcwd()
training_data1 = pd.read_csv('D:/pythoncode/shopeeMiniProject/shopeeMiniProject/training_data1.csv')
svd_fea = pd.read_csv('D:/pythoncode/shopeeMiniProject/shopeeMiniProject/svd_fea.csv')
training_data2 = pd.merge(training_data1, svd_fea, how='left', on=['userid'])
training_data2.to_csv('training_data2.csv', index=False)

# add view log features
training_data1 = pd.read_csv('./training_data1.csv')
view_log_fea = pd.read_csv('./view_pre30days_feature.csv')
# add one column convert unix epoch time to date
training_data1['voucher_received_date'] = pd.to_datetime(training_data1['voucher_received_time'],unit='s')
training_data1['voucher_received_date'] = training_data1['voucher_received_date'].apply(lambda x: str(x)[:10])
testing_data['voucher_received_date'] = pd.to_datetime(testing_data['voucher_received_time'],unit='s')
testing_data['voucher_received_date'] = testing_data['voucher_received_date'].apply(lambda x: str(x)[:10])
training_data3 = pd.merge(training_data1, view_log_fea, how='left', on=['userid','voucher_received_date'])
testing_data = pd.merge(testing_data, view_log_fea, how='left', on=['userid','voucher_received_date'])
training_data3.to_csv('training_data3.csv', index=False)

# add trans features
training_data4 = pd.merge(training_data3, merged_user_fea, how='left', on=['userid'])
merged_voucher_fea=merged_voucher_fea.rename(columns = {'voucher_code_used':'voucher_code_received'})
training_data4 = pd.merge(training_data4, merged_voucher_fea, how='left', on=['voucher_code_received'])
merged_promotion_fea=merged_promotion_fea.rename(columns = {'promotionid_used':'promotionid_received'})
training_data4 = pd.merge(training_data4, merged_promotion_fea, how='left', on=['promotionid_received'])
training_data4.to_csv('training_data4.csv', index=False)

# add trans fea for testing data
merged_user_fea = pd.read_csv('./trans_user_fea.csv')
testing_data = pd.merge(testing_data, merged_user_fea, how='left', on=['userid'])
merged_voucher_fea = pd.read_csv('./trans_voucher_fea.csv')
merged_voucher_fea=merged_voucher_fea.rename(columns = {'voucher_code_used':'voucher_code_received'})
testing_data = pd.merge(testing_data, merged_voucher_fea, how='left', on=['voucher_code_received'])
merged_promotion_fea = pd.read_csv('./trans_promotion_fea.csv')
merged_promotion_fea=merged_promotion_fea.rename(columns = {'promotionid_used':'promotionid_received'})
testing_data = pd.merge(testing_data, merged_promotion_fea, how='left', on=['promotionid_received'])
merge_user_voucher = pd.read_csv('./trans_user_voucher_fea.csv')
merge_user_voucher=merge_user_voucher.rename(columns = {'voucher_code_used':'voucher_code_received'})
testing_data = pd.merge(testing_data, merge_user_voucher, how='left', on=['userid','voucher_code_received']) 
merge_user_promotion = pd.read_csv('./trans_user_promotion_fea.csv')
merge_user_promotion=merge_user_promotion.rename(columns = {'promotionid_used':'promotionid_received'})
testing_data = pd.merge(testing_data, merge_user_promotion, how='left', on=['userid','promotionid_received']) 
merge_user_voucher_promotion = pd.read_csv('./trans_user_voucher_promotion_fea.csv')
merge_user_voucher_promotion=merge_user_voucher_promotion.rename(columns = {'voucher_code_used':'voucher_code_received','promotionid_used':'promotionid_received'})
testing_data = pd.merge(testing_data, merge_user_voucher_promotion, how='left', on=['userid','voucher_code_received','promotionid_received'])


# add more trans features including user voucher relation,  user promotion relation and user voucher promotion relation 
merge_user_voucher=merge_user_voucher.rename(columns = {'voucher_code_used':'voucher_code_received'})
training_data5 = pd.merge(training_data4, merge_user_voucher, how='left', on=['userid','voucher_code_received'])
merge_user_promotion=merge_user_promotion.rename(columns = {'promotionid_used':'promotionid_received'})
training_data5 = pd.merge(training_data5, merge_user_promotion, how='left', on=['userid','promotionid_received'])
merge_user_voucher_promotion=merge_user_voucher_promotion.rename(columns = {'voucher_code_used':'voucher_code_received','promotionid_used':'promotionid_received'})
training_data5 = pd.merge(training_data5, merge_user_voucher_promotion, how='left', on=['userid','voucher_code_received','promotionid_received'])
training_data5.to_csv('training_data5.csv', index=False)

# add svd fea to traning_data 5
training_data6 = pd.merge(training_data5, svd_fea, how='left', on=['userid'])
training_data6.to_csv('training_data6.csv', index=False)

# add active sess 0-30 mean, median, max, std.
training_data6 = pd.merge(training_data6, added_feas, how='left', on=['userid', 'promotionid_received','voucher_code_received','voucher_received_time'])
training_data6.to_csv('training_data6.csv', index=False)
testing_data = pd.merge(testing_data, added_feas, how='left', on=['userid', 'promotionid_received','voucher_code_received','voucher_received_time'])


# drop previous generated 20 svd features
training_data6 = training_data6.drop(['col1', 'col2', 'col3', 'col4', 'col5',
                                      'col6', 'col7', 'col8', 'col9', 'col10',
                                      'col11', 'col12', 'col13', 'col14', 'col15',
                                      'col16', 'col17', 'col18', 'col19', 'col20',], axis=1)
training_data6.to_csv('training_data6.csv', index=False)

# add 198 new likes features
training_data6 = pd.read_csv('./training_data6.csv')
training_data7 = pd.merge(training_data6, merged_like_fea, how='left', on=['userid'])
training_data7.to_csv('training_data7.csv', index=False)
# training_data7 shape (710078, 442)
merged_like_fea = pd.read_csv('./user_likes_fea.csv')
testing_data = pd.merge(testing_data, merged_like_fea, how='left', on=['userid'])

# add voucher day actions
voucher_day_action = pd.read_csv('./voucher_day_actions.csv')
training_data7 = pd.merge(training_data7, voucher_day_action, how='left', on=['userid','voucher_received_date'])
training_data7.to_csv('training_data7.csv', index=False)
testing_data = pd.merge(testing_data, voucher_day_action, how='left', on=['userid','voucher_received_date'])
testing_data.to_csv('testing_data.csv', index=False)

# total 438 features
# training_data7 shape (710078, 448)
# testing_data shape (78903, 443)
    