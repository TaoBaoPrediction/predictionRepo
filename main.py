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
from sklearn.metrics import accuracy_score
import pickle



def preprocessing():
    df_train = pd.read_csv("./data/data_format1/train_format1.csv")
    df_user_info = pd.read_csv("./data/data_format1/user_info_format1.csv")
    df_user_log = pd.read_csv("./data/data_format1/user_log_format1.csv")
    df_test = pd.read_csv("./data/data_format1/test_format1.csv")

    # Dealing with missing data
    # df_train contains no missing data

    # df_user_info
    # Fill NAN with most frequent value
    df_user_info['age_range'].replace(-1, np.nan, inplace=True)
    df_user_info['gender'].replace(-1, np.nan, inplace=True)
    df_user_info = df_user_info.fillna(df_user_info.mode().iloc[0])
    # print(df_user_info.info())


    # df_user_log
    # Remove（Brand_id == Nan)
    # df_user_log.dropna(inplace=True)

    # Feature Engineering
    # Merge train and user information
    df_train = pd.merge(df_train, df_user_info, on="user_id", how="left")
    print(df_train.head())

    # Grab [user_id, seller_id, item_id] triplets
    log_temp = df_user_log.groupby([df_user_log["user_id"], df_user_log["seller_id"]]).count().reset_index()[
        ["user_id", "seller_id", "item_id"]]
    log_temp.rename(columns={"seller_id": "merchant_id", "item_id": "total_logs"}, inplace=True)
    print(log_temp.head())
    df_train = pd.merge(df_train, log_temp, on=["user_id", "merchant_id"], how="left")
    print(df_train.head())

    # Grab unique id
    log_id_temp = df_user_log.groupby([df_user_log["user_id"], df_user_log["seller_id"],
                                  df_user_log["item_id"]]).count().reset_index()[["user_id", "seller_id", "item_id"]]
    log_id = log_id_temp.groupby(
        [log_id_temp["user_id"], log_id_temp["seller_id"]]).count().reset_index()
    log_id.rename(columns={"seller_id": "merchant_id", "item_id": "unique_item_ids"}, inplace=True)
    df_train = pd.merge(df_train, log_id, on=["user_id", "merchant_id"], how="left")
    print(df_train.head())

    # Categories
    categories_temp = df_user_log.groupby([df_user_log["user_id"], df_user_log["seller_id"],
                                           df_user_log["cat_id"]]).count().reset_index()[["user_id", "seller_id", "cat_id"]]
    categories_temp1 = categories_temp.groupby(
        [categories_temp["user_id"], categories_temp["seller_id"]]).count().reset_index()
    categories_temp1.rename(columns={"seller_id": "merchant_id", "cat_id": "categories"}, inplace=True)
    df_train = pd.merge(df_train, categories_temp1, on=["user_id", "merchant_id"], how="left")
    print(df_train.head())

    # Browse_days
    browse_days_temp = df_user_log.groupby([df_user_log["user_id"], df_user_log["seller_id"],
                                            df_user_log["time_stamp"]]).count().reset_index()[["user_id", "seller_id", "time_stamp"]]
    browse_days_temp1 = browse_days_temp.groupby(
        [browse_days_temp["user_id"], browse_days_temp["seller_id"]]).count().reset_index()

    browse_days_temp1.rename(columns={"seller_id": "merchant_id", "time_stamp": "browse_days"}, inplace=True)
    df_train = pd.merge(df_train, browse_days_temp1, on=["user_id", "merchant_id"], how="left")
    print(df_train.head())

    # One_click
    one_clicks_temp = df_user_log.groupby([df_user_log["user_id"], df_user_log["seller_id"], df_user_log["action_type"]]
                                          ).count().reset_index()[["user_id", "seller_id", "action_type", "item_id"]]
    one_clicks_temp.rename(columns={"seller_id": "merchant_id", "item_id": "times"}, inplace=True)
    one_clicks_temp["one_clicks"] = one_clicks_temp["action_type"] == 0
    one_clicks_temp["one_clicks"] = one_clicks_temp["one_clicks"] * one_clicks_temp["times"]
    one_clicks_temp["shopping_carts"] = one_clicks_temp["action_type"] == 1
    one_clicks_temp["shopping_carts"] = one_clicks_temp["shopping_carts"] * one_clicks_temp["times"]
    one_clicks_temp["purchase_times"] = one_clicks_temp["action_type"] == 2
    one_clicks_temp["purchase_times"] = one_clicks_temp["purchase_times"] * one_clicks_temp["times"]
    one_clicks_temp["favourite_times"] = one_clicks_temp["action_type"] == 3
    one_clicks_temp["favourite_times"] = one_clicks_temp["favourite_times"] * one_clicks_temp["times"]
    four_features = one_clicks_temp.groupby([one_clicks_temp["user_id"],
                                             one_clicks_temp["merchant_id"]]).sum().reset_index()
    four_features = four_features.drop(["action_type", "times"], axis=1)
    df_train = pd.merge(df_train, four_features, on=["user_id", "merchant_id"], how="left")
    print(df_train.head())
    df_train.info()
    df_train = df_train.fillna(method='ffill')

    # df_test
    df_test = pd.merge(df_test, df_user_info, on="user_id", how="left")
    df_test = pd.merge(df_test, log_temp, on=["user_id", "merchant_id"], how="left")
    df_test = pd.merge(df_test, log_id, on=["user_id", "merchant_id"], how="left")
    df_test = pd.merge(df_test, categories_temp1, on=["user_id", "merchant_id"], how="left")
    df_test = pd.merge(df_test, browse_days_temp1, on=["user_id", "merchant_id"], how="left")
    df_test = pd.merge(df_test, four_features, on=["user_id", "merchant_id"], how="left")
    df_test = df_test.fillna(method='bfill')
    df_test = df_test.fillna(method='ffill')

    return df_train, df_test

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
    X = test.drop(['user_id', 'merchant_id', 'prob'],axis = 1)
    choose = ["user_id", "merchant_id", "prob"]
    prob = model.predict_proba(X)
    test['prob'] = pd.Series(prob[:, 1])
    test = test[choose]
    test.to_csv('submission.csv', index=False)

def main():
    # df_train, df_test = preprocessing()
    # df_train.to_pickle("df_train.pkl")
    # df_test.to_pickle("df_test.pkl")

    # Model Training
    df_train = pd.read_pickle("df_train.pkl")
    df_test = pd.read_pickle("df_test.pkl")
    Train_x = df_train.drop(['user_id', 'merchant_id', 'label'], axis=1)
    Train_y = df_train['label']
    model = xgBoost(Train_x, Train_y)

    # Prediction
    predict_to_csv(model, df_test)

if __name__ == "__main__":
    main()
