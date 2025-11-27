#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 2025

Modified to use XGBoost instead of PLS Regression.
"""

import pandas as pd
import numpy as np
# 新增: 导入 XGBoost 和多输出包装器
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
# -------------------------------------------
from scipy.stats import pearsonr
from joblib import dump
from pathlib import Path


def get_mse(y_gt, y_predict):
    # Compute mean square error
    MSE = np.mean((y_gt - y_predict) ** 2, axis=0)
    return MSE.tolist()


def get_ccc(y_gt, y_predict):
    # Compute canonical correlation coefficient
    CCC = []
    for i in range(y_gt.shape[1]):
        A = y_gt[:, i]
        B = y_predict[:, i]
        pearson = pearsonr(A, B)
        # 防止除零错误
        if (A.std() ** 2 + B.std() ** 2 + (A.mean() - B.mean()) ** 2) == 0:
            c = 0
        else:
            c = ((2 * pearson[0] * A.std() * B.std()) /
                 (A.std() ** 2 + B.std() ** 2 + (A.mean() - B.mean()) ** 2))
        CCC.append(c)
    return CCC


# if FULL_FEATURES=False (exclude jawline) resulting dimensionality -> 1276
# if FULL_FEATURES=True (all 68 landmarks) resulting dimensionality -> 2278
def load_and_split_data(full_features=False, path_data='../data/', file_name='AFEW-VAset.csv'):
    # load datasets and frontalization weights
    print('Data loading...')
    # ATTENTION. You need to run extract_features.py in order to generate features
    # before running this script!
    feature_file = f'{path_data}{Path(file_name).stem}_features_fullfeatures={full_features}.npy'

    try:
        features = np.load(feature_file)
    except FileNotFoundError:
        print(f"Error: Feature file not found at {feature_file}")
        print("Please run extract_features.py first.")
        raise

    df_data = pd.read_csv(f'{path_data}{file_name}')

    # split data
    np.random.seed(1)
    subjets = df_data['Subject'].unique()
    np.random.shuffle(subjets)

    # 70% train, 20% validation, 10% testing subjects
    subjects_train, subjects_val, subjects_test = np.split(
        subjets,
        [int(.7 * len(subjets)), int(.9 * len(subjets))]
    )

    # subjects to indices
    indx_train = list(df_data['Subject'].isin(subjects_train))
    indx_val = list(df_data['Subject'].isin(subjects_val))
    indx_test = list(df_data['Subject'].isin(subjects_test))

    # split features
    features_train = features[indx_train, :]
    features_val = features[indx_val, :]
    features_test = features[indx_test, :]

    # split annotations
    avi_train = df_data.iloc[indx_train, 5:8].values.astype(np.float16)
    avi_val = df_data.iloc[indx_val, 5:8].values.astype(np.float16)
    avi_test = df_data.iloc[indx_test, 5:8].values.astype(np.float16)

    print("Features Max:", np.max(features))
    print("Features Min:", np.min(features))

    return {
        "train": (features_train, avi_train),
        "val": (features_val, avi_val),
        "test": (features_test, avi_test)
    }


def train_model(X_train, y_train):
    print("Initializing XGBoost model...")
    # XGBoost 参数设置
    # n_estimators: 树的数量，越多通常越准，但也越慢
    # learning_rate: 学习率，越低越稳，但需要更多的树
    # max_depth: 树的深度，控制模型复杂度，太深容易过拟合
    xgb_estimator = XGBRegressor(
        n_estimators=500,  # 可以尝试调整为 1000
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,  # 每次只用80%的数据训练，防止过拟合
        colsample_bytree=0.8,  # 每次只用80%的特征，防止过拟合
        n_jobs=-1,  # 使用所有CPU核心
        objective='reg:squarederror',
        random_state=42
    )

    # 包装为多输出回归器 (因为我们要同时预测 Arousal, Valence, Intensity)
    model = MultiOutputRegressor(xgb_estimator)

    print("Training started (this might take a while)...")
    model.fit(X_train, y_train)
    print("Training finished.")

    return model


if __name__ == "__main__":
    FULL_FEATURES = False
    # XGBoost 不需要 components 参数，这里保留变量名只是为了逻辑清晰
    MODEL_TYPE = "XGBoost_v1"

    PATH_MODELS = '../models/'
    PATH_DATA = '../data/'
    FILE_NAME = 'Morphset.csv'

    # 确保输出目录存在
    Path(PATH_MODELS).mkdir(parents=True, exist_ok=True)

    data = load_and_split_data(full_features=FULL_FEATURES, path_data=PATH_DATA, file_name=FILE_NAME)
    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']

    print(f'Training with {MODEL_TYPE}...')
    model = train_model(X_train, y_train)

    print('\n--- Evaluation ---')
    y_predict_val = model.predict(X_val)
    print('Validation MSE=', get_mse(y_val, y_predict_val))
    print('Validation CCC=', get_ccc(y_val, y_predict_val))

    y_predict_test = model.predict(X_test)
    print('Test MSE=', get_mse(y_test, y_predict_test))
    print('Test CCC=', get_ccc(y_test, y_predict_test))

    # save model
    # 我们将 components 字段设置为字符串，这样 emotions_dlib.py 加载时打印出来会显示 "Model components: XGBoost..."
    output = {'model': model, 'full_features': FULL_FEATURES, 'components': 'XGBoost'}

    save_path = f'{PATH_MODELS}model_emotion_from{Path(FILE_NAME).stem}_{MODEL_TYPE}_fullfeatures={FULL_FEATURES}.joblib'
    dump(output, save_path)
    print(f'\nModel saved to: {save_path}')