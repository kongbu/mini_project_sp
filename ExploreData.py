# explore the data
import pandas as pd
import time
import datetime as DT

# training data
training_data = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/training.csv')
print(training_data.shape) #(710078, 9)
training_data.head()
training_data.userid.nunique() #279825
training_data.promotionid_received.nunique() #92
training_data.voucher_code_received.nunique() #92
# check if promotionid and voucher_code one to one matach
promotionid = training_data.promotionid_received.unique()
sum_voucherUnique = 0
for i in range(92):
    sum_voucherUnique = sum_voucherUnique + training_data[training_data.promotionid_received ==  promotionid[i]].voucher_code_received.nunique()
print sum_voucherUnique #92, which means they are one to one match
# positive pecentage
training_data['used?'].sum()/710078.0           #3.3%   23453 
training_data['repurchase_15?'].sum()/710078.0  #22.7%  161137
training_data['repurchase_30?'].sum()/710078.0  #35.2%  250262
training_data['repurchase_60?'].sum()/710078.0  #49.2%  349362
training_data['repurchase_90?'].sum()/710078.0  #55.8%  396140
# time period
training_data.voucher_received_time.min() #1484820308
training_data.voucher_received_time.max() #1502861172
# convert unix epoch time
time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1484820308)) #2017-01-19 18:05:08
time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1502861172)) #2017-08-16 13:26:12
# add one column convert unix epoch time to date
training_data['voucher_received_date'] = pd.to_datetime(training_data['voucher_received_time'],unit='s')
training_data['voucher_received_date'] = training_data['voucher_received_date'].apply(lambda x: str(x)[:10])

# testing data
testing_data = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/predict.csv')
print(testing_data.shape) #(78903, 4)
testing_data.userid.nunique() #78903
testing_data.promotionid_received.nunique() #4
testing_data.voucher_code_received.nunique() #4, one match one
# time period
testing_data.voucher_received_time.min() #1502861173
testing_data.voucher_received_time.max() #1502867977
# convert unix epoch time
time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1502861173)) #2017-08-16 13:26:13
time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1502867977)) #2017-08-16 15:19:37
# compare sets
len(set(testing_data.userid.unique()).intersection(set(training_data.userid.unique()))) #65903 in common
set(testing_data.promotionid_received.unique()).intersection(set(training_data.promotionid_received.unique())) #2 in common
set(testing_data.voucher_code_received.unique()).intersection(set(training_data.voucher_code_received.unique())) # 2 in common

#transaction data
trans_data = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/transactions_MY.csv')
trans_data['promotionid_used'] = trans_data['promotionid_used'].astype(object)
print(trans_data.shape) #(3460223, 7)
trans_data.head()
trans_data.userid.nunique() #292809
trans_data.shopid.nunique() #41249
trans_data.voucher_code_used.nunique() #11049
trans_data.promotionid_used.nunique() #5519
# check one to one match, some promotionid has many voucher_code which is different from training and testing
promotionid = trans_data.promotionid_used.unique()
sum_voucherUnique = 0
for i in range(92):
    num_unique_voucherid = trans_data[trans_data.promotionid_used ==  promotionid[i]].voucher_code_used.nunique()
    sum_voucherUnique = sum_voucherUnique + num_unique_voucherid
    if num_unique_voucherid != 1:
        print promotionid[i]
print sum_voucherUnique
# time period
trans_data.order_time.min() #1432709432
trans_data.order_time.max() #1506362011
# convert unix epoch time
time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1432709432)) #2015-05-27 14:50:32
time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1506362011)) #2017-09-26 01:53:31
# compare sets
len(set(trans_data.userid.unique()).intersection(set(training_data.userid.unique()))) #279803 in common, 22 users in train not in trans
len(set(trans_data.voucher_code_used.unique()).intersection(set(training_data.voucher_code_received.unique()))) #92
len(set(trans_data.promotionid_used.unique()).intersection(set(training_data.promotionid_received.unique()))) #92

# user profile data
usr_profile = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/user_profiles_MY.csv')
usr_profile.shape #(513052, 7)
usr_profile.gender.nunique() #4
#min max register time
usr_profile.registration_time.min() #2015-05-27 10:11:10
usr_profile.registration_time.max() #2017-07-14 23:41:35
#calculate age range
now = pd.Timestamp(DT.datetime.now())
usr_profile['dob'] = pd.to_datetime(usr_profile['birthday'], format='%Y-%m-%d')
usr_profile['age'] = (now - usr_profile['dob']).astype('<m8[Y]')
# some wrong birthday used, 17 age <0
usr_profile['age'].min() #-21
usr_profile['age'].max() #115
usr_profile.age.value_counts() #most of them 20-30

# like data
like_data = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/likes.csv')
like_data.shape #(11901222, 4)
like_data.head()
like_data.userid.nunique() #262345
like_data.itemid.nunique() #2706236
# time period
time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(like_data.ctime.min())) #2015-06-02 22:42:47
time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(like_data.ctime.max())) #2017-09-25 01:58:05

# view log data
view_log_0 = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/view_log_0.csv')
view_log_0.shape #(1097235, 4)
view_log_0.head() 
view_log_0.event_name.nunique() #5, 
#'trackGenericScroll', 'trackGenericSearchPageView','trackGenericClick', 'trackSearchFilterApplied', 'addItemToCart', nan
view_log_0.userid.nunique() #292825

view_log_1 = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/view_log_1.csv')
view_log_1.shape #(1054142, 4)
view_log_1.head() 
view_log_1.event_name.nunique() #5, 
#'trackGenericScroll', 'trackGenericSearchPageView','trackGenericClick', 'trackSearchFilterApplied', 'addItemToCart',nan
view_log_1.userid.nunique() #292825

# voucher_distribution_active_date,  active sessions
active_sess = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/voucher_distribution_active_date.csv')
active_sess.shape #(788981, 36)
active_sess.head()
active_sess.userid.nunique() #292825

#Voucher_mechanics.csv  voucher information
voucher_info = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/Voucher_mechanics.csv')
voucher_info.shape #(94, 3)
voucher_info.head()