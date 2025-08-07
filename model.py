import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder

# Loading the dataset to create a DataFrame
df = pd.read_csv('D:\90 days DSA Goat\ML journey\Build-an-MLOps-pipeline-\Indians screentime\Indian_Kids_Screen_Time.csv')


print(df.head(), "\n")
df = df.drop("Health_Impacts", axis='columns')
print(df)
# print("\n", df.isna().sum())


# Plotting Bar Graph for two features of Dataset

# plt.bar(df['Age'], df['Avg_Daily_Screen_Time_hr'])
# plt.xlabel('Age')
# plt.ylabel('Avg_Daily_Screen_Time_hr')
# plt.show()

# Encoding the categorical columns using dummies

df = pd.get_dummies(df, columns=['Gender', 'Primary_Device'], dtype=int)
df = df.drop(['Gender_Female','Primary_Device_Tablet'], axis='columns')
# df.to_csv('columns.csv')

print("\n New DataFrame without dummy columns \n", df)

# Assigning features and target values to X and y respectively. (Feature Engineering)
X = df.drop('Urban_or_Rural', axis='columns')
y = df['Urban_or_Rural']

print("Features \n", X)
print("Target \n", y)


# Using GridSearchCV find the best performance model for prediction
model_params = {
    # 'svm':{

    # "model":svm.SVC(gamma='auto'),
    # "params":{
    #     "C":[1,5,10],
    #     "kernel": ['rbf', 'poly']
    #     }
    # },
    'random_forest': {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [1,5,20]
        }
    },

    'logistic_regression': {
        "model": LogisticRegression(solver='liblinear', multi_class='auto'),
        "params": {
            "C": [1,5,20]
        }
    },

    'k_nearest_neighbors': {
        "model": KNeighborsClassifier(),
        "params": {
            'n_neighbors': np.arange(1,5),
            'p': np.arange(1,3)
        }
    }
    
}

scores = []

for model_name, mp in model_params.items():
    print(f"\n🔄 Testing {model_name}...")
    
    clf = GridSearchCV(
        mp['model'], 
        mp['params'], 
        cv=3,  # Reduced from 5 to 3 folds for faster execution
        return_train_score=False,  # Don't return train scores to save time
        n_jobs=-1,  # Use all available cores for parallel processing
        verbose=1  # Show progress during fitting
    )
    
    print("🚀 Starting to fit features and target from Dataset...")
    clf.fit(X, y)
    print("✅ Dataset fitting complete!")
    
    scores.append({
        "model": model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_
    })

df1 = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df1)
