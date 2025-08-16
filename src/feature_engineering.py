import pandas as pd
import matplotlib.pyplot as plt
import os
from data_preprocessing import data_preprocessing


def feature_engineering(df, dummy_columns:list, target_column:str, dummy_trap:list):
    
    '''Encoding the categorical columns using dummies'''

    df = pd.get_dummies(df, columns=dummy_columns, dtype=int)
    df = df.drop(dummy_trap, axis='columns')

    ''' Assigning features and target values to X and y respectively. (Feature Engineering) '''

    X = df.drop(target_column, axis='columns')
    y = df[target_column]

    return X, y

df = data_preprocessing('D:\90 days DSA Goat\ML journey\Build-an-MLOps-pipeline-\Indians screentime\Indian_Kids_Screen_Time.csv', "Health_Impacts")

print(feature_engineering(df,['Gender', 'Primary_Device'] ,['Urban_or_Rural'], ['Gender_Female', 'Primary_Device_Tablet']))