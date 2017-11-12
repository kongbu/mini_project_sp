# extract fea in likes.csv

# 1. like number,  like unique #item, like unique days, 
# like number, item, unique days for each month
# unlike same
like_data = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/likes.csv')
# add one column convert unix epoch time to date
like_data['like_date'] = pd.to_datetime(like_data['ctime'],unit='s')
like_data['like_date'] = like_data['like_date'].apply(lambda x: str(x)[:10])
# like feas
like_subset =  like_data.loc[like_data['status']==1]
total_likes = like_subset.groupby(['userid']).size().rename('num_total_likes').to_frame().reset_index()
total_likeitems = like_subset.groupby(['userid']).itemid.nunique().rename('num_total_likeitems').to_frame().reset_index()
total_likedays = like_subset.groupby(['userid']).like_date.nunique().rename('num_total_like_days').to_frame().reset_index()
# likes in each month
like_subset['order_month'] = like_subset['like_date'].apply(lambda x: str(x)[:7])
total_likes_permonth = like_subset.groupby(['userid', 'order_month']).size().rename('num_likes_permonth').to_frame().reset_index()
pivot_likes_permonth =  pd.pivot_table(total_likes_permonth, values='num_likes_permonth', index=['userid'], columns='order_month').add_prefix('likes_in_').reset_index()
pivot_likes_permonth = pivot_likes_permonth.fillna(value=0)
pivot_likes_permonth['mean_likespermonth'] = pivot_likes_permonth.iloc[:, 1:-1].mean(axis=1)
pivot_likes_permonth['std_likespermonth'] = pivot_likes_permonth.iloc[:, 1:-1].std(axis=1)
pivot_likes_permonth['max_likespermonth'] = pivot_likes_permonth.iloc[:, 1:-1].max(axis=1)
pivot_likes_permonth['median_likespermonth'] = pivot_likes_permonth.iloc[:, 1:-1].median(axis=1)
total_likeitems_permonth = like_subset.groupby(['userid', 'order_month']).itemid.nunique().rename('num_likeitems_permonth').to_frame().reset_index()
pivot_likeitems_permonth =  pd.pivot_table(total_likeitems_permonth, values='num_likeitems_permonth', index=['userid'], columns='order_month').add_prefix('likeitems_in_').reset_index()
pivot_likeitems_permonth = pivot_likeitems_permonth.fillna(value=0)
pivot_likeitems_permonth['mean_likeitemspermonth'] = pivot_likeitems_permonth.iloc[:, 1:-1].mean(axis=1)
pivot_likeitems_permonth['std_likeitemspermonth'] = pivot_likeitems_permonth.iloc[:, 1:-1].std(axis=1)
pivot_likeitems_permonth['max_likeitemspermonth'] = pivot_likeitems_permonth.iloc[:, 1:-1].max(axis=1)
pivot_likeitems_permonth['median_likeitemspermonth'] = pivot_likeitems_permonth.iloc[:, 1:-1].median(axis=1)
total_likedays_permonth = like_subset.groupby(['userid', 'order_month']).like_date.nunique().rename('num_likedays_permonth').to_frame().reset_index()
pivot_likedays_permonth =  pd.pivot_table(total_likedays_permonth, values='num_likedays_permonth', index=['userid'], columns='order_month').add_prefix('likedays_in_').reset_index()
pivot_likedays_permonth = pivot_likedays_permonth.fillna(value=0)
pivot_likedays_permonth['mean_likedayspermonth'] = pivot_likedays_permonth.iloc[:, 1:-1].mean(axis=1)
pivot_likedays_permonth['std_likedayspermonth'] = pivot_likedays_permonth.iloc[:, 1:-1].std(axis=1)
pivot_likedays_permonth['max_likedayspermonth'] = pivot_likedays_permonth.iloc[:, 1:-1].max(axis=1)
pivot_likedays_permonth['median_likedayspermonth'] = pivot_likedays_permonth.iloc[:, 1:-1].median(axis=1)
# unlike feas
unlike_subset =  like_data.loc[like_data['status']==0]
total_unlikes = unlike_subset.groupby(['userid']).size().rename('num_total_unlikes').to_frame().reset_index()
total_unlikeitems = unlike_subset.groupby(['userid']).itemid.nunique().rename('num_total_unlikeitems').to_frame().reset_index()
total_unlikedays = unlike_subset.groupby(['userid']).like_date.nunique().rename('num_total_unlike_days').to_frame().reset_index()
# unlikes in each month
unlike_subset['order_month'] = unlike_subset['like_date'].apply(lambda x: str(x)[:7])
total_unlikes_permonth = unlike_subset.groupby(['userid', 'order_month']).size().rename('num_unlikes_permonth').to_frame().reset_index()
pivot_unlikes_permonth =  pd.pivot_table(total_unlikes_permonth, values='num_unlikes_permonth', index=['userid'], columns='order_month').add_prefix('unlikes_in_').reset_index()
pivot_unlikes_permonth = pivot_unlikes_permonth.fillna(value=0)
pivot_unlikes_permonth['mean_unlikespermonth'] = pivot_unlikes_permonth.iloc[:, 1:-1].mean(axis=1)
pivot_unlikes_permonth['std_unlikespermonth'] = pivot_unlikes_permonth.iloc[:, 1:-1].std(axis=1)
pivot_unlikes_permonth['max_unlikespermonth'] = pivot_unlikes_permonth.iloc[:, 1:-1].max(axis=1)
pivot_unlikes_permonth['median_unlikespermonth'] = pivot_likes_permonth.iloc[:, 1:-1].median(axis=1)
total_unlikeitems_permonth = unlike_subset.groupby(['userid', 'order_month']).itemid.nunique().rename('num_unlikeitems_permonth').to_frame().reset_index()
pivot_unlikeitems_permonth =  pd.pivot_table(total_unlikeitems_permonth, values='num_unlikeitems_permonth', index=['userid'], columns='order_month').add_prefix('unlikeitems_in_').reset_index()
pivot_unlikeitems_permonth = pivot_unlikeitems_permonth.fillna(value=0)
pivot_unlikeitems_permonth['mean_unlikeitemspermonth'] = pivot_unlikeitems_permonth.iloc[:, 1:-1].mean(axis=1)
pivot_unlikeitems_permonth['std_unlikeitemspermonth'] = pivot_unlikeitems_permonth.iloc[:, 1:-1].std(axis=1)
pivot_unlikeitems_permonth['max_unlikeitemspermonth'] = pivot_unlikeitems_permonth.iloc[:, 1:-1].max(axis=1)
pivot_unlikeitems_permonth['median_unlikeitemspermonth'] = pivot_unlikeitems_permonth.iloc[:, 1:-1].median(axis=1)
total_unlikedays_permonth = unlike_subset.groupby(['userid', 'order_month']).like_date.nunique().rename('num_unlikedays_permonth').to_frame().reset_index()
pivot_unlikedays_permonth =  pd.pivot_table(total_unlikedays_permonth, values='num_unlikedays_permonth', index=['userid'], columns='order_month').add_prefix('unlikedays_in_').reset_index()
pivot_unlikedays_permonth = pivot_unlikedays_permonth.fillna(value=0)
pivot_unlikedays_permonth['mean_unlikedayspermonth'] = pivot_unlikedays_permonth.iloc[:, 1:-1].mean(axis=1)
pivot_unlikedays_permonth['std_unlikedayspermonth'] = pivot_unlikedays_permonth.iloc[:, 1:-1].std(axis=1)
pivot_unlikedays_permonth['max_unlikedayspermonth'] = pivot_unlikedays_permonth.iloc[:, 1:-1].max(axis=1)
pivot_unlikedays_permonth['median_unlikedayspermonth'] = pivot_unlikedays_permonth.iloc[:, 1:-1].median(axis=1)
# merge like feas
merged_like_fea = total_likes.merge(total_likeitems,on=['userid']).merge(total_likedays,on=['userid']).merge(pivot_likes_permonth,on=['userid']).merge(pivot_likeitems_permonth,on=['userid']).merge(pivot_likedays_permonth,on=['userid']).merge(total_unlikes,on=['userid']).merge(total_unlikeitems,on=['userid']).merge(total_unlikedays,on=['userid']).merge(pivot_unlikes_permonth,on=['userid']).merge(pivot_unlikeitems_permonth,on=['userid']).merge(pivot_unlikedays_permonth,on=['userid'])
merged_like_fea.to_csv('user_likes_fea.csv', index=False)  

# svd feature
users = like_data.userid.unique()
uid_dict = {k: v for v,k in enumerate(users)}
items = like_data.itemid.unique()
item_dict = {k: v for v,k in enumerate(items)}

row_ind = []
col_ind = []
value = []
for index, row in like_data.iterrows():
    row_ind.append(uid_dict[row['userid']])
    col_ind.append(item_dict[row['itemid']])
    value.append(1)
print 'the number of users is %f:' % (len(users))
print 'the number of items is %f:' %(len(items))
print "the sparsity of the whole user item matrix is %f" %(float(len(value))/(len(users)*len(items)))
row_ind = np.array(row_ind)
col_ind = np.array(col_ind)
value = np.array(value)
data_csc = csc_matrix((value, (row_ind, col_ind)), shape=(len(users), len(items)))
ut, s, vt = sparsesvd(data_csc, 20)
ut3 = np.vstack((users,ut))
ut4=ut3.transpose()
aa = ['col']*21
bb = range(21)
col_names=["{}{}".format(a_, b_) for a_, b_ in zip(aa, bb)]
ut_df = pd.DataFrame(ut4,columns=col_names)
ut_df.rename(columns={'col0': 'userid'}, inplace=True)
ut_df['userid'] = ut_df['userid'].astype(int)
ut_df.to_csv('svd_fea.csv', index=False)