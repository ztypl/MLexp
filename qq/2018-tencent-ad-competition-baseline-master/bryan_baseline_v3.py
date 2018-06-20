# coding=utf-8
# @author:bryan
# blog: https://blog.csdn.net/bryan__
# github: https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import math
import numpy as np



def get_user_feature():
    if os.path.exists('../data/userFeature.csv'):
        user_feature=pd.read_csv('../data/userFeature.csv')
    else:
        userFeature_data = []
        with open('../data/userFeature.data', 'r') as f:
            cnt = 0
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                    print(i)
                if i % 1000000 == 0:
                    user_feature = pd.DataFrame(userFeature_data)
                    user_feature.to_csv('../data/userFeature_' + str(cnt) + '.csv', index=False)
                    cnt += 1
                    del userFeature_data, user_feature
                    userFeature_data = []
            user_feature = pd.DataFrame(userFeature_data)
            user_feature.to_csv('../data/userFeature_' + str(cnt) + '.csv', index=False)
            del userFeature_data, user_feature
            user_feature = pd.concat(
                [pd.read_csv('../data/userFeature_' + str(i) + '.csv') for i in range(cnt + 1)]).reset_index(drop=True)
            user_feature.to_csv('../data/userFeature.csv', index=False)
    return user_feature

def get_data():
    if os.path.exists('../data/data.csv'):
        return pd.read_csv('../data/data.csv')
    else:
        ad_feature = pd.read_csv('../data/adFeature.csv')
        train=pd.read_csv('../data/train.csv')
        predict=pd.read_csv('../data/test2.csv')
        train.loc[train['label']==-1,'label']=0
        predict['label']=-1
        user_feature=get_user_feature()

        user_feature['house'] = user_feature['house'].fillna(0).astype('int')

        data=pd.concat([train,predict])
        data=pd.merge(data,ad_feature,on='aid',how='left')
        data=pd.merge(data,user_feature,on='uid',how='left')
        data=data.fillna('-1')
        del user_feature
        return data

def batch_predict(data,index):
    one_hot_feature=['LBS','gender','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType']
    vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    for ct_i in [0, 1, 2, 3, 4]:
        data['ct_%d' % ct_i] = data['ct'].apply(lambda x: int(str(ct_i) in x.split()))

    del data['ct']

    for os_i in [0, 1, 2]:
        data['os_%d' % os_i] = data['os'].apply(lambda x: int(str(os_i) in x.split()))

    del data['os']

    for ms_i in {0, 10, 11, 12, 13, 14, 15, 2, 3, 5, 6, 8, 9}:
        data['ms_%d' % ms_i] = data['marriageStatus'].fillna("-1").apply(lambda x: int(str(ms_i) in x.split()))

    del data['marriageStatus']

    train=data[data.label!=-1]
    train_y=train.pop('label')
    test=data[data.label==-1]
    res=test[['aid','uid']]
    test=test.drop('label',axis=1)
    enc = OneHotEncoder()

    ini_cols = ['creativeSize','age','carrier','consumptionAbility','education','house'] \
        + ["ct_%d" % ct_i for ct_i in[0, 1, 2, 3, 4]] \
        + ["os_%d" % os_i for os_i in [0, 1, 2]] \
        + ['ms_%d' % ms_i for ms_i in {0, 10, 11, 12, 13, 14, 15, 2, 3, 5, 6, 8, 9}]

    train_x=train[ini_cols]
    test_x=test[ini_cols]

    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a=enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x= sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
        print(feature+' finish')
    print('one-hot prepared !')

    cv=CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature])
        train_a = cv.transform(train[feature])
        test_a = cv.transform(test[feature])
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
        print(feature + ' finish')
    print('cv prepared !')
    del data
    return LGB_predict(train_x, train_y, test_x, res,index)

def LGB_predict(train_x,train_y,test_x,res,index):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1500, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    res['score'+str(index)] = clf.predict_proba(test_x)[:,1]
    res['score'+str(index)] = res['score'+str(index)].apply(lambda x: float('%.6f' % x))
    print(str(index)+' predict finish!')
    gc.collect()
    res=res.reset_index(drop=True)
    return res['score'+str(index)]

#数据分片处理，对每片分别训练预测，然后求平均
data=get_data()
train=data[data['label']!=-1]
test=data[data['label']==-1]
del data
predict=pd.read_csv('../data/test2.csv')
cnt=20
size = math.ceil(len(train) / cnt)
result=[]
for i in range(cnt):
    start = size * i
    end = (i + 1) * size if (i + 1) * size < len(train) else len(train)
    slice = train[start:end]
    result.append(batch_predict(pd.concat([slice,test]),i))
    gc.collect()

result=pd.concat(result,axis=1)
result['score']=np.mean(result,axis=1)
result=result.reset_index(drop=True)
result=pd.concat([predict[['aid','uid']].reset_index(drop=True),result['score']],axis=1)
result[['aid','uid','score']].to_csv('../data/submission.csv', index=False)
os.system('zip ../data/baseline.zip ../data/submission.csv')