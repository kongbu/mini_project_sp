import pandas as pd
import time
import xgboost as xgb
from sklearn.metrics import average_precision_score
# training data time period: 2017-01-19 18:05:08 - 2017-08-16 13:26:12 about 6 month
# partition by time, use 2 hours as validation, from 2017-08-16 11:26:12
training_data1 = pd.read_csv('./training_data1.csv')
training_data2 = pd.read_csv('./training_data2.csv')
training_data4 = pd.read_csv('./training_data4.csv')
training_data6 = pd.read_csv('./training_data6.csv')
training_data7 = pd.read_csv('./training_data7.csv')
partition_time = int(time.mktime(time.strptime('2017-08-16 11:26:12', '%Y-%m-%d %H:%M:%S')))
train = training_data7[training_data7.voucher_received_time < partition_time]
validation = training_data7[training_data7.voucher_received_time >= partition_time]
train.shape  #684988
validation.shape #25090
# positive pecentage
train['used?'].sum()/684988.0           #3.3%  
train['repurchase_15?'].sum()/684988.0  #22.5% 
train['repurchase_30?'].sum()/684988.0  #34.9%
train['repurchase_60?'].sum()/684988.0  #49.2%
train['repurchase_90?'].sum()/684988.0  #56.0%
train.isnull().sum() #31.9% gender missing, 52.3% age missing
validation['used?'].sum()/25090.0           #2.9%
validation['repurchase_15?'].sum()/25090.0  #27.1%
validation['repurchase_30?'].sum()/25090.0  #43.7%
validation['repurchase_60?'].sum()/25090.0  #49.3%
validation['repurchase_90?'].sum()/25090.0  #49.3%
validation.isnull().sum() #42.8% gender missing, 55.3% age missing

used_train = train['used?']
repurchase_15_train = train['repurchase_15?']
repurchase_30_train = train['repurchase_30?']
repurchase_60_train = train['repurchase_60?']
repurchase_90_train = train['repurchase_90?']
used_valid = validation['used?']
repurchase_15_valid = validation['repurchase_15?']
repurchase_30_valid = validation['repurchase_30?']
repurchase_60_valid = validation['repurchase_60?']
repurchase_90_valid = validation['repurchase_90?']
train = train.drop(['userid','promotionid_received','voucher_code_received','voucher_received_time','voucher_received_date',
'used?','repurchase_15?','repurchase_30?','repurchase_60?','repurchase_90?'],axis=1)
#,'gender','age'
validation = validation.drop(['userid','promotionid_received','voucher_code_received','voucher_received_time','voucher_received_date',
'used?','repurchase_15?','repurchase_30?','repurchase_60?','repurchase_90?'],axis=1)
#,'gender','age'

params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": 0.08,
        #"eta": 0.05,
        "max_depth": 6,
        "subsample": 0.75,
        "min_chil_weight":50,
        "silent": 1,
        "nthread": 4,
    }
nround = 200
#nround = 300
#predict if used
#params['scale_pos_weight'] = 100/3.3
xgtrain = xgb.DMatrix(train.values, used_train.values)
xgvalid = xgb.DMatrix(validation.values, used_valid.values)
xgb_model = xgb.train(params, xgtrain, nround)
# save out model
xgb_model.save_model('used_model_train6_2')
predict_res = xgb_model.predict(xgb.DMatrix(validation.values))
#AUC
score = average_precision_score(used_valid.values, predict_res)
print('auc of predict if used: {:.6f}'.format(score)) #base: 0.083521, add gender and age: 0.092905
                                                      #train4: 0.194446 #train6 0.05 7 300 100%
                                                      #train7: 100%
#train6 model 0.08 6 200 100%
#bst = xgb.Booster({'nthread':4}, model_file = 'used_model_train6')
#predict_res_1 = bst.predict(xgb.DMatrix(validation.values))

with open('out.txt','w') as f:
    print >> f, 'auc of predict if used:', score

#predict repurchase_15
xgtrain = xgb.DMatrix(train.values, repurchase_15_train.values)
xgvalid = xgb.DMatrix(validation.values, repurchase_15_valid.values)
params1 = params
params1['scale_pos_weight'] = 100/22.5
xgb_model = xgb.train(params1, xgtrain, nround)
xgb_model.save_model('re15_model_train6_2')
predict_res = xgb_model.predict(xgb.DMatrix(validation.values))
#AUC
score = average_precision_score(repurchase_15_valid.values, predict_res)
print('auc of repurchase_15: {:.6f}'.format(score)) #base: 0.376060, add gender and age: 0.376740 
                                                    #train4: 0.375188 #train6 0.05 7 300 0.385768 #train7: 0.370390
with open('out1.txt','w') as f:
    print >> f, 'auc of predict if used:', score
#train6 model 0.08 6 200 0.373466
#bst = xgb.Booster({'nthread':4}, model_file = 're15_model_train6')
#predict_res_1 = bst.predict(xgb.DMatrix(validation.values))

#predict repurchase_30
xgtrain = xgb.DMatrix(train.values, repurchase_30_train.values)
xgvalid = xgb.DMatrix(validation.values, repurchase_30_valid.values)
xgb_model = xgb.train(params, xgtrain, nround)
xgb_model.save_model('re30_model_train6_2')
predict_res = xgb_model.predict(xgb.DMatrix(validation.values))
#AUC
score = average_precision_score(repurchase_30_valid.values, predict_res)
print('auc of repurchase_30: {:.6f}'.format(score)) #base: 0.563166, add gender and age: 0.562894
                                                    #0.628535 #train6 0.05 7 300 0.612136 #train7: 0.629324
#train6 model 0.08 6 200 0.638619
#bst = xgb.Booster({'nthread':4}, model_file = 're30_model_train6')
#predict_res_1 = bst.predict(xgb.DMatrix(validation.values))

with open('out2.txt','w') as f:
    print >> f, 'auc of predict if used:', score
#predict repurchase_60
xgtrain = xgb.DMatrix(train.values, repurchase_60_train.values)
xgvalid = xgb.DMatrix(validation.values, repurchase_60_valid.values)
xgb_model = xgb.train(params, xgtrain, nround)
xgb_model.save_model('re60_model_train6_2')
predict_res = xgb_model.predict(xgb.DMatrix(validation.values))
#AUC
score = average_precision_score(repurchase_60_valid.values, predict_res)
print('auc of repurchase_60: {:.6f}'.format(score)) #base: 0.619861, add gender and age: 0.622719
                                                    #0.887228 #train6 0.05 7 300 0.882999 #train7: 0.889767
#train6 model 0.08 6 200 0.891242
#bst = xgb.Booster({'nthread':4}, model_file = 're60_model_train6')
#predict_res_1 = bst.predict(xgb.DMatrix(validation.values))

with open('out3.txt','w') as f:
    print >> f, 'auc of predict if used:', score
#predict repurchase_90
xgtrain = xgb.DMatrix(train.values, repurchase_90_train.values)
xgvalid = xgb.DMatrix(validation.values, repurchase_90_valid.values)
xgb_model = xgb.train(params, xgtrain, nround)
xgb_model.save_model('re90_model_train6_2')
predict_res = xgb_model.predict(xgb.DMatrix(validation.values))
#AUC
score = average_precision_score(repurchase_90_valid.values, predict_res)
print('auc of repurchase_90: {:.6f}'.format(score)) #base: 0.616958, add gender and age: 0.619703 train4:0.980057
                                                    #train6 0.05 7 300 0.981999 #train7: 0.982639
#train6 model 0.08 6 200 0.980429
#bst = xgb.Booster({'nthread':4}, model_file = 're90_model_train6')
#predict_res_1 = bst.predict(xgb.DMatrix(validation.values))
with open('out4.txt','w') as f:
    print >> f, 'auc of predict if used:', score

fea_importance = pd.DataFrame(xgb_model.get_score(importance_type='gain').items(), columns=['feature','importance']).sort_values('importance', ascending=False)
fea_importance.head()


#training use whole training data
train = pd.read_csv('./training_data7.csv')
used_train = train['used?']
repurchase_15_train = train['repurchase_15?']
repurchase_30_train = train['repurchase_30?']
repurchase_60_train = train['repurchase_60?']
repurchase_90_train = train['repurchase_90?']
train = train.drop(['userid','promotionid_received','voucher_code_received','voucher_received_time','voucher_received_date',
'used?','repurchase_15?','repurchase_30?','repurchase_60?','repurchase_90?'],axis=1)

params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": 0.08,
        #"eta": 0.05,
        "max_depth": 6,
        "subsample": 0.75,
        "min_chil_weight":50,
        "silent": 1,
        "nthread": 4,
    }
nround = 200
#predict if use voucher
xgtrain = xgb.DMatrix(train.values, used_train.values)
xgb_model = xgb.train(params, xgtrain, nround)
# save out model
xgb_model.save_model('used_model_train7')
predict_res = xgb_model.predict(xgb.DMatrix(train.values))
#AUC
score = average_precision_score(used_train.values, predict_res)
print('auc of predict if used: {:.6f}'.format(score))
with open('out.txt','w') as f:
    print >> f, 'auc of predict if used:', score
#predict repurchase_15
xgtrain = xgb.DMatrix(train.values, repurchase_15_train.values)
params1 = params
params1['scale_pos_weight'] = 100/22.7
xgb_model = xgb.train(params1, xgtrain, nround)
xgb_model.save_model('re15_model_train7')
predict_res = xgb_model.predict(xgb.DMatrix(train.values))
#AUC
score = average_precision_score(repurchase_15_train.values, predict_res)
print('auc of repurchase_15: {:.6f}'.format(score)) 
with open('out1.txt','w') as f:
    print >> f, 'auc of predict if used:', score
#predict repurchase_30
xgtrain = xgb.DMatrix(train.values, repurchase_30_train.values)
xgb_model = xgb.train(params, xgtrain, nround)
xgb_model.save_model('re30_model_train7')
predict_res = xgb_model.predict(xgb.DMatrix(train.values))
#AUC
score = average_precision_score(repurchase_30_train.values, predict_res)
print('auc of repurchase_30: {:.6f}'.format(score)) 
with open('out2.txt','w') as f:
    print >> f, 'auc of predict if used:', score
#predict repurchase_60
xgtrain = xgb.DMatrix(train.values, repurchase_60_train.values)
xgb_model = xgb.train(params, xgtrain, nround)
xgb_model.save_model('re60_model_train7')
predict_res = xgb_model.predict(xgb.DMatrix(train.values))
#AUC
score = average_precision_score(repurchase_60_train.values, predict_res)
print('auc of repurchase_60: {:.6f}'.format(score))
with open('out3.txt','w') as f:
    print >> f, 'auc of predict if used:', score
#predict repurchase_90
xgtrain = xgb.DMatrix(train.values, repurchase_90_train.values)
xgb_model = xgb.train(params, xgtrain, nround)
xgb_model.save_model('re90_model_train7')
predict_res = xgb_model.predict(xgb.DMatrix(train.values))
#AUC
score = average_precision_score(repurchase_90_train.values, predict_res)
print('auc of repurchase_90: {:.6f}'.format(score)) 
with open('out4.txt','w') as f:
    print >> f, 'auc of predict if used:', score

# write the submission file
submission = pd.read_csv('D:/Projects/ShopeeMiniProject/Data/predict.csv')
testing_data = pd.read_csv('./testing_data.csv')
testing_data = testing_data.drop(['userid','promotionid_received','voucher_code_received','voucher_received_time','voucher_received_date'],axis=1)
bst = xgb.Booster({'nthread':4}, model_file = 'used_model_train7')
predict_res_1 = bst.predict(xgb.DMatrix(testing_data.values))
submission['used?'] = predict_res_1
bst = xgb.Booster({'nthread':4}, model_file = 're15_model_train7')
predict_res_1 = bst.predict(xgb.DMatrix(testing_data.values))
submission['repurchase_15?'] = predict_res_1
bst = xgb.Booster({'nthread':4}, model_file = 're30_model_train7')
predict_res_1 = bst.predict(xgb.DMatrix(testing_data.values))
submission['repurchase_30?'] = predict_res_1
bst = xgb.Booster({'nthread':4}, model_file = 're60_model_train7')
predict_res_1 = bst.predict(xgb.DMatrix(testing_data.values))
submission['repurchase_60?'] = predict_res_1
bst = xgb.Booster({'nthread':4}, model_file = 're90_model_train7')
predict_res_1 = bst.predict(xgb.DMatrix(testing_data.values))
submission['repurchase_90?'] = predict_res_1
submission.to_csv('submission.csv', index=False)