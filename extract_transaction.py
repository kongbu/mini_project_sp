import pandas as pd

def extract_transaction_feature():
    trans = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/transactions_MY.csv')
    # add one column convert unix epoch time to date
    trans['order_date'] = pd.to_datetime(trans['order_time'],unit='s')
    trans['order_date'] = trans['order_date'].apply(lambda x: str(x)[:10])
    trans.shape # (3460223, 8)
    trans.userid.nunique() # 292809
    trans.voucher_code_used.nunique() # 11049
    trans.promotionid_used.nunique() # 5519
    trans.order_date.min() # 2015-05-27
    trans.order_date.max() # 2017-09-25

    # each user's feature
    total_trans = trans.groupby(['userid']).size().rename('num_total_trans').to_frame().reset_index()
    total_shop = trans.groupby(['userid']).shopid.nunique().rename('num_total_shop').to_frame().reset_index()
    total_voucher = trans.groupby(['userid']).voucher_code_used.nunique().rename('num_total_voucher').to_frame().reset_index()
    total_promotion = trans.groupby(['userid']).promotionid_used.nunique().rename('num_total_promotion').to_frame().reset_index()
    total_shopping_days = trans.groupby(['userid']).order_date.nunique().rename('num_total_shoppingday').to_frame().reset_index()
    total_payment = trans.groupby(['userid']).total_price.sum().rename('num_total_payment').to_frame().reset_index()
    # each month number of trans, shopping days, voucher use and promotion use
    trans['order_month'] = trans['order_date'].apply(lambda x: str(x)[:7])
    total_trans_permonth = trans.groupby(['userid', 'order_month']).size().rename('num_trans_permonth').to_frame().reset_index()
    pivot_trans_permonth =  pd.pivot_table(total_trans_permonth, values='num_trans_permonth', index=['userid'], columns='order_month').reset_index()
    pivot_trans_permonth = pivot_trans_permonth.fillna(value=0)
    pivot_trans_permonth['mean_transpermonth'] = pivot_trans_permonth.iloc[:, 1:-1].mean(axis=1)
    pivot_trans_permonth['std_transpermonth'] = pivot_trans_permonth.iloc[:, 1:-1].std(axis=1)
    pivot_trans_permonth['max_transpermonth'] = pivot_trans_permonth.iloc[:, 1:-1].max(axis=1)
    pivot_trans_permonth['median_transpermonth'] = pivot_trans_permonth.iloc[:, 1:-1].median(axis=1)
    total_shopping_days_permonth = trans.groupby(['userid', 'order_month']).order_date.nunique().rename('num_shopping_days_permonth').to_frame().reset_index()
    pivot_shopping_days_permonth =  pd.pivot_table(total_shopping_days_permonth, values='num_shopping_days_permonth', index=['userid'], columns='order_month').reset_index()
    pivot_shopping_days_permonth = pivot_shopping_days_permonth.fillna(value=0)
    pivot_shopping_days_permonth['mean_shopdaypermonth'] = pivot_shopping_days_permonth.iloc[:, 1:-1].mean(axis=1) 
    pivot_shopping_days_permonth['std_shopdaypermonth'] = pivot_shopping_days_permonth.iloc[:, 1:-1].std(axis=1)
    pivot_shopping_days_permonth['max_shopdaypermonth'] = pivot_shopping_days_permonth.iloc[:, 1:-1].max(axis=1)
    pivot_shopping_days_permonth['median_shopdaypermonth'] = pivot_shopping_days_permonth.iloc[:, 1:-1].median(axis=1)
    total_voucheruse_permonth = trans.groupby(['userid', 'order_month']).voucher_code_used.nunique().rename('num_voucheruse_permonth').to_frame().reset_index()
    pivot_voucheruse_permonth =  pd.pivot_table(total_voucheruse_permonth, values='num_voucheruse_permonth', index=['userid'], columns='order_month').reset_index()
    pivot_voucheruse_permonth = pivot_voucheruse_permonth.fillna(value=0)
    pivot_voucheruse_permonth['mean_voucherpermonth'] = pivot_voucheruse_permonth.iloc[:, 1:-1].mean(axis=1) 
    pivot_voucheruse_permonth['std_voucherpermonth'] = pivot_voucheruse_permonth.iloc[:, 1:-1].std(axis=1)
    pivot_voucheruse_permonth['max_voucherpermonth'] = pivot_voucheruse_permonth.iloc[:, 1:-1].max(axis=1)
    pivot_voucheruse_permonth['median_voucherpermonth'] = pivot_voucheruse_permonth.iloc[:, 1:-1].median(axis=1)
    total_promotionuse_permonth = trans.groupby(['userid', 'order_month']).promotionid_used.nunique().rename('num_promotionuse_permonth').to_frame().reset_index()
    pivot_promotionuse_permonth =  pd.pivot_table(total_promotionuse_permonth, values='num_promotionuse_permonth', index=['userid'], columns='order_month').reset_index()
    pivot_promotionuse_permonth = pivot_promotionuse_permonth.fillna(value=0)
    pivot_promotionuse_permonth['mean_promotionpermonth'] = pivot_promotionuse_permonth.iloc[:, 1:-1].mean(axis=1) 
    pivot_promotionuse_permonth['std_promotionpermonth'] = pivot_promotionuse_permonth.iloc[:, 1:-1].std(axis=1)
    pivot_promotionuse_permonth['max_promotionpermonth'] = pivot_promotionuse_permonth.iloc[:, 1:-1].max(axis=1)
    pivot_promotionuse_permonth['median_promotionpermonth'] = pivot_promotionuse_permonth.iloc[:, 1:-1].median(axis=1)
    # merge user feature
    merged_user_fea = total_trans.merge(total_shop,on=['userid']).merge(total_voucher,on=['userid']).merge(total_promotion,on=['userid']).merge(total_shopping_days,on=['userid']).merge(total_payment,on=['userid']).merge(pivot_trans_permonth,on=['userid']).merge(pivot_shopping_days_permonth,on=['userid']).merge(pivot_voucheruse_permonth,on=['userid']).merge(pivot_promotionuse_permonth,on=['userid'])
    merged_user_fea.to_csv('trans_user_fea.csv', index=False)    

    ## voucher feature
    total_voucher_trans = trans.groupby(['voucher_code_used']).size().rename('num_voucher_trans').to_frame().reset_index()
    total_voucher_shop = trans.groupby(['voucher_code_used']).shopid.nunique().rename('num_voucher_shop').to_frame().reset_index()
    total_voucher_user = trans.groupby(['voucher_code_used']).userid.nunique().rename('num_voucher_user').to_frame().reset_index()
    total_voucher_payment = trans.groupby(['voucher_code_used']).total_price.sum().rename('num_voucher_payment').to_frame().reset_index()
    merged_voucher_fea = total_voucher_trans.merge(total_voucher_shop,on=['voucher_code_used']).merge(total_voucher_user,on=['voucher_code_used']).merge(total_voucher_payment,on=['voucher_code_used'])
    merged_voucher_fea.to_csv('trans_voucher_fea.csv', index=False) 
    ## promotion fea
    total_promotion_trans = trans.groupby(['promotionid_used']).size().rename('num_promotion_trans').to_frame().reset_index()
    total_promotion_shop = trans.groupby(['promotionid_used']).shopid.nunique().rename('num_promotion_shop').to_frame().reset_index()
    total_promotion_user = trans.groupby(['promotionid_used']).userid.nunique().rename('num_promotion_user').to_frame().reset_index()
    total_promotion_payment = trans.groupby(['promotionid_used']).total_price.sum().rename('num_promotion_payment').to_frame().reset_index()
    merged_promotion_fea = total_promotion_trans.merge(total_promotion_shop,on=['promotionid_used']).merge(total_promotion_user,on=['promotionid_used']).merge(total_promotion_payment,on=['promotionid_used'])
    merged_promotion_fea.to_csv('trans_promotion_fea.csv', index=False) 

    ## user voucher relation 
    user_voucher = trans.groupby(['userid','voucher_code_used']).size().rename('num_user_voucher_used').to_frame().reset_index()
    user_voucher_shops = trans.groupby(['userid','voucher_code_used']).shopid.nunique().rename('num_user_voucher_shops').to_frame().reset_index()
    user_voucher_days = trans.groupby(['userid','voucher_code_used']).order_date.nunique().rename('num_user_voucher_days').to_frame().reset_index()
    merge_user_voucher = user_voucher.merge(user_voucher_shops,on=['userid','voucher_code_used']).merge(user_voucher_days,on=['userid','voucher_code_used'])
    merge_user_voucher.to_csv('trans_user_voucher_fea.csv', index=False)
    ## user promotion relation 
    user_promotion = trans.groupby(['userid','promotionid_used']).size().rename('num_user_promotion_used').to_frame().reset_index()
    user_promotion_shops = trans.groupby(['userid','promotionid_used']).shopid.nunique().rename('num_user_promotion_shops').to_frame().reset_index()
    user_promotion_days = trans.groupby(['userid','promotionid_used']).order_date.nunique().rename('num_user_promotion_days').to_frame().reset_index()
    merge_user_promotion = user_promotion.merge(user_promotion_shops,on=['userid','promotionid_used']).merge(user_promotion_days,on=['userid','promotionid_used'])
    merge_user_promotion.to_csv('trans_user_promotion_fea.csv', index=False)
    ## user voucher promotion relation 
    user_voucher_promotion = trans.groupby(['userid','voucher_code_used','promotionid_used']).size().rename('num_user_voucher_promotion_used').to_frame().reset_index()
    user_voucher_promotion_shops = trans.groupby(['userid','voucher_code_used','promotionid_used']).shopid.nunique().rename('num_user_voucher_promotion_shops').to_frame().reset_index()
    user_voucher_promotion_days = trans.groupby(['userid','voucher_code_used','promotionid_used']).order_date.nunique().rename('num_user_voucher_promotion_days').to_frame().reset_index()
    merge_user_voucher_promotion = user_voucher_promotion.merge(user_voucher_promotion_shops,on=['userid','voucher_code_used','promotionid_used']).merge(user_voucher_promotion_days,on=['userid','voucher_code_used','promotionid_used'])
    merge_user_voucher_promotion.to_csv('trans_user_voucher_promotion_fea.csv', index=False)










if __name__ == "__main__":
    extract_transaction_feature()