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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc,roc_curve
import pickle
import lightgbm as lgb

def xgBoost(Train_x, Train_y):
    X_train, X_test, y_train, y_test = train_test_split(Train_x, Train_y, test_size=0.25, random_state=10)

    model = xgb.XGBClassifier(
        max_depth=8,
        n_estimators=2000,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.3,
        seed=17373331
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
    submission = pd.read_csv(f'./data/data_format1/test_format1.csv')
    prob = model.predict_proba(test)
    submission['prob'] = pd.Series(prob[:, 1])
    # submission.drop(['origin'], axis=1, inplace=True)
    print(submission)
    print('begin save')
    submission.to_csv('submission.csv', index=False)


def lightGBM(Train_x, Train_y):
    X_train, X_valid, y_train, y_valid = train_test_split(Train_x, Train_y, test_size=0.15, random_state=10)
    model = lgb.LGBMClassifier(
        max_depth=10,  # 8
        n_estimators=2000,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.3,
        seed=42
    )
    model.fit(
        X_train,
        y_train,
        eval_metric='auc',
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=True,
        early_stopping_rounds=10
    )

    X_valid = model.predict(X_valid)
    auc_score = roc_auc_score(y_valid, X_valid)

    print("Training Accuracy:", auc_score)

    return model


def main():

    # df_train.to_csv('df_train.csv', index=False)
    # df_test.to_csv('df_test.csv', index=False)
    # df_train.to_pickle("df_train.pkl")
    # df_test.to_pickle("df_test.pkl")

    # Model Training
    df_train = pd.read_csv("train_df.csv")
    df_test = pd.read_csv("test_df.csv")
    Train_x = df_train.drop(['user_id', 'merchant_id', 'label'], axis=1)
    Train_y = df_train['label']
    # model = xgBoost(Train_x, Train_y)
    model = lightGBM(Train_x, Train_y)

    # Prediction
    predict_to_csv(model, df_test.drop(['user_id', 'merchant_id'], axis=1))


if __name__ == "__main__":
    main()
