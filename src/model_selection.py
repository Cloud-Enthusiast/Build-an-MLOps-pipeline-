import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import yaml
from logger import logging
from feature_engineering import feature_engineering
from data_preprocessing import data_preprocessing

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

def load_params(params_path:str):
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.debug('Parameters retrieved from %s', params_path)
        return params

# Using GridSearchCV find the best performance model for prediction

def model_selection(X,y):
    params = load_params('params.yaml')
    
    model_params = {
        'rfc' : RandomForestClassifier(),
        'logreg': LogisticRegression(),
        'knn' : KNeighborsClassifier()
    }

    scores = []

    for model_name, mp in model_params.items():
        print(f"\n🔄 Testing {model_name}... on parameters {params[model_name]}")
        
        clf = GridSearchCV(
            estimator=mp,
            param_grid= params[model_name], 
            cv=3,  # Reduced from 5 to 3 folds for faster execution
            return_train_score=False,  # Don't return train scores to save time
            n_jobs=-1,  # Use all available cores for parallel processing
            verbose=1  # Show progress during fitting
        )
        
        print("🚀 Starting to fit features and target from Dataset...")
        clf.fit(X, y)
        print("✅ Dataset fitting complete!\n")
        
        scores.append({
            "model": model_name,
            "best_score": clf.best_score_,
            "best_params": clf.best_params_
        })

    df1 = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    return df1

print(load_params('D:\90 days DSA Goat\ML journey\Build-an-MLOps-pipeline-\params.yaml'))

df = data_preprocessing('D:\90 days DSA Goat\ML journey\Build-an-MLOps-pipeline-\Indians screentime\Indian_Kids_Screen_Time.csv', "Health_Impacts")

fe = feature_engineering(df,['Gender', 'Primary_Device'] ,['Urban_or_Rural'], ['Gender_Female', 'Primary_Device_Tablet'])

print(model_selection(fe[0], fe[1]))