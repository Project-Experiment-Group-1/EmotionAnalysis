#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:10:20 2021

@author: Vasileios Vonikakis
@email: bbonik@gmail.com

This script loads geometrical facial features (which have been pre-extracted
previously by the extract_features.py script) and trains a Partial Leasts 
Squares facial expression analysis model. The combination of these 2 scripts, 
is also included in the notebook file Extract_features_and_train_model.ipynb, 
in a more explanatory way. In order for this script to run successfuly, you
need first to extract and save the features, either by running 
extract_features.py or Extract_features_and_train_model.ipynb.
"""


import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr
from joblib import dump
from pathlib import Path


def get_mse(y_gt,y_predict):
    # Compute mean square error
    MSE = np.mean((y_gt - y_predict)**2, axis=0)
#    return np.sqrt(MSE).tolist()
    return MSE.tolist()

def get_ccc(y_gt,y_predict):
    # Compute canonical correlation coefficient
    CCC=[]
    for i in range(y_gt.shape[1]):
        A=y_gt[:,i]
        B=y_predict[:,i]
        pearson = pearsonr(A, B)
        c = ((2 * pearson[0] * A.std() * B.std()) / 
             ( A.std()**2 + B.std()**2 + (A.mean() - B.mean())**2 ))
        CCC.append(c)
    return CCC

# if FULL_FEATURES=False (exclude jawline) resulting dimensionality -> 1276
# if FULL_FEATURES=True (all 68 landmarks) resulting dimensionality -> 2278
def load_and_split_data(full_features=False, path_data='../data/', file_name='AFEW-VAset.csv'):
    # load datasets and frontalization weights
    print('Data loading...')
    # ATTENTION. You need to run extract_features.py in order to generate features
    # before running this script!
    features = np.load(f'{path_data}{Path(file_name).stem}_features_fullfeatures={full_features}.npy')
    df_data = pd.read_csv(f'{path_data}{file_name}')

    # split data
    np.random.seed(1)
    subjets = df_data['Subject'].unique()
    np.random.shuffle(subjets)

    # 70% train, 20% validation, 10% testing subjects
    subjects_train, subjects_val, subjects_test = np.split(
        subjets,
        [int(.7*len(subjets)), int(.9*len(subjets))]
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
    avi_train = df_data.iloc[indx_train,5:8].values.astype(np.float16)
    avi_val = df_data.iloc[indx_val,5:8].values.astype(np.float16)
    avi_test = df_data.iloc[indx_test,5:8].values.astype(np.float16)

    print("Features Max:", np.max(features))
    print("Features Min:", np.min(features))

    return {
        "train": (features_train, avi_train),
        "val": (features_val, avi_val),
        "test": (features_test, avi_test)
    }

def train_model(X_train, y_train, n_components):
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    return pls

if __name__ == "__main__":
    FULL_FEATURES = True
    COMPONENTS = 29
    PATH_MODELS = '../models/'
    PATH_DATA = '../data/'
    FILE_NAME = 'AFEW-VAset.csv'

    data = load_and_split_data(full_features=FULL_FEATURES, path_data=PATH_DATA, file_name=FILE_NAME)
    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']

    print('PLS regression...')
    model = train_model(X_train, y_train, n_components=COMPONENTS)

    y_predict_val = model.predict(X_val)
    print('Validation MSE=', get_mse(y_val, y_predict_val))
    print('Validation CCC=', get_ccc(y_val, y_predict_val))

    y_predict_test = model.predict(X_test)
    print('Test MSE=', get_mse(y_test, y_predict_test))
    print('Test CCC=', get_ccc(y_test, y_predict_test))

    # save model
    output = {'model': model, 'full_features': FULL_FEATURES, 'components': COMPONENTS}
    dump(output, f'{PATH_MODELS}model_emotion_from{Path(FILE_NAME).stem}_pls={COMPONENTS}_fullfeatures={FULL_FEATURES}.joblib')
