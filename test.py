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
import warnings
warnings.filterwarnings('ignore')

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

# CORRECTED: Fixed all syntax errors and optimized model_params dictionary
model_params = {
    # 'svm': {
    #     "model": svm.SVC(gamma='auto'),
    #     "params": {
    #         "C": [1,3],  # Reduced parameter space for faster execution
    #         "kernel": ['poly']  # Start with just rbf kernel
    #     }
    # },
    'random_forest': {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [1, 10, 50]  # Reduced and more reasonable values
        }
    },
    'logistic_regression': {
        "model": LogisticRegression(solver='liblinear', multi_class='auto', random_state=42),
        "params": {
            "C": [1, 10, 20]  # Reduced parameter space
        }
    },
    'k_nearest_neighbors': {
        "model": KNeighborsClassifier(),
        "params": {
            'n_neighbors': [3, 15],  # Reduced from np.arange(1,5) for better performance
            'p': [1, 2]  # Reduced from np.arange(1,3)
        }
    }
}

# IMPROVED: Added progress tracking and performance optimization
scores = []

print("=" * 60)
print("STARTING MODEL TRAINING AND EVALUATION")
print("=" * 60)

for model_name, mp in model_params.items():
    print(f"\n🔄 Testing {model_name}...")
    
    # OPTIMIZED: GridSearchCV with performance improvements
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
    
    # Store results with additional information
    scores.append({
        "model": model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_
    })
    
    # Display immediate results for each model
    print(f"   📊 Best Score: {clf.best_score_:.4f}")
    print(f"   ⚙️  Best Params: {clf.best_params_}")

# ENHANCED: Better results display
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

df1 = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Sort by best score (descending)
df1_sorted = df1.sort_values('best_score', ascending=False).reset_index(drop=True)

print("\n📈 RANKED RESULTS (Best to Worst):")
print("-" * 40)
for idx, row in df1_sorted.iterrows():
    rank = idx + 1
    print(f"{rank}. {row['model']:20} | Score: {row['best_score']:.4f}")
    print(f"   Best Parameters: {row['best_params']}")
    print()

print(f"🏆 WINNER: {df1_sorted.iloc[0]['model']} with score {df1_sorted.iloc[0]['best_score']:.4f}")

# Display the complete results DataFrame
print("\n📋 COMPLETE RESULTS TABLE:")
print(df1_sorted.to_string(index=False))
