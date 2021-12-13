"""
    天猫复购预测挑战 url：https://tianchi.aliyun.com/competition/entrance/231576/introduction
    Course: 大数据分析B 2021
    Date: 11th Dec 2021
    Group Members: 王子牧 彭杰 张耕纶
    Python version: 3.7
"""

# TODO:
"""
    1. Preprocessing
    2. Feature Engineering
        2.1 从 user_info和user_log里提取出信息
    3. 可视化 - Report/Slide 图
    4. Model 
        3.1 xgBoost
        3.2 lightGBM
        3.3 MLP
    5. 结果保存
"""

# Load all the libraries
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
#import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc,roc_curve
import pickle
#import lightgbm as lgb
from sklearn.neural_network import MLPClassifier 

def xgBoost(Train_x, Train_y):
    X_train, X_test, y_train, y_test = train_test_split(Train_x, Train_y, test_size=0.25, random_state=10)

    model = xgb.XGBClassifier(
        max_depth=8,
        n_estimators=500,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.5,
        eta=0.3,
        seed=42
    )

    model.fit(
        X_train,
        y_train,
        eval_metric='auc',
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True,
        early_stopping_rounds=10
    )

    return model

def predict_to_csv(model, test):
    submission = pd.read_csv(f'sample_submission.csv')
    prob = model.predict(test)
    print(prob)
    submission['prob'] = pd.Series(prob[:, 1])
    # submission.drop(['origin'], axis=1, inplace=True)
    print(submission)
    print('begin save')
    submission.to_csv('submission.csv', index=False)


def lightGBM(Train_x, Train_y):
    X_train, X_valid, y_train, y_valid = train_test_split(Train_x, Train_y, test_size=0.15, random_state=10)
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    X_valid = model.predict(X_valid)
    auc_score = roc_auc_score(y_valid, X_valid)

    print("Training Accuracy:", auc_score)

    return model
def pre_process(df):
    #user_id,merchant_id,label,age_range,gender,total_logs,unique_item_ids,categories,browse_days,one_clicks,shopping_carts,purchase_times,favourite_times
    for column in df:
        if column in ['age_range','browse_days','one_clicks','purchase_times','favourite_times','total_logs']:
            df[column] = (df[column]-df[column].min()) / (df[column].max()-df[column].min())
    return df            

        
def mlpRegressor(Train_x, Train_y):
    X_train, X_valid, y_train, y_valid = train_test_split(Train_x, Train_y, test_size=0.15, random_state=10)
    model = MLPClassifier(hidden_layer_sizes=(100,), 
              activation="logistic", 
              solver="sgd", 
              alpha=0.0001, 
              batch_size="auto", 
              learning_rate="adaptive", 
              learning_rate_init=0.001, 
              power_t=0.5, 
              max_iter=200, 
              shuffle=True, 
              random_state=None, 
              tol=0.0001, 
              verbose=True, 
              warm_start=False, 
              momentum=0.9, 
              nesterovs_momentum=True, 
              early_stopping=False, 
              validation_fraction=0.1, 
              beta_1=0.9, 
              beta_2=0.999, 
              epsilon=1e-08)
    model.fit(X_train, y_train)
    predict = model.predict(X_valid)
    auc_score = roc_auc_score(y_valid,predict)
    print("Training Accuracy:", auc_score)

    return model
def main():

    # df_train.to_csv('df_train.csv', index=False)
    # df_test.to_csv('df_test.csv', index=False)
    # df_train.to_pickle("df_train.pkl")
    # df_test.to_pickle("df_test.pkl")

    # Model Training
    df_train = pd.read_csv("df_train.csv")
    df_test = pd.read_csv("df_test.csv")
    Train_x = df_train.drop(['user_id', 'merchant_id', 'label'], axis=1)
    Norm_x = pre_process(Train_x)
    print(Norm_x)
    Train_y = df_train['label']
    model = mlpRegressor(Train_x, Train_y)
    # model = lightGBM(Train_x, Train_y)

    Norm_test = pre_process(df_test.drop(['user_id', 'merchant_id','prob'],axis = 1))
    # Prediction
    predict_to_csv(model,Norm_test)


if __name__ == "__main__":
    main()
