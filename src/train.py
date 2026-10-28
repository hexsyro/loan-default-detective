from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import joblib


x_train=pd.read_csv('data/processed/x_train_scaled.csv')
y_train=pd.read_csv("data/processed/Y_train.csv").squeeze()

MODELS = {
    "LR": LogisticRegression(C=0.1, max_iter=50000, class_weight="balanced"),
    "RFC": RandomForestClassifier(n_estimators=300, min_samples_leaf=25,
                                  max_features="sqrt", class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42),
    "DTC": DecisionTreeClassifier(min_samples_leaf=50,
                                  class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=150, weights="distance", n_jobs=-1),
    "GNB": GaussianNB(var_smoothing=1e-6),
    "SVM": LinearSVC(C=1.0,class_weight="balanced",random_state=42)}

def main(x_train,y_train, MODELS):
    for name,MODEL in MODELS.items():
        print(f"Training {name}...", flush=True)
        MODEL.fit(x_train,y_train)
        joblib.dump(MODEL, f'models/{name}.joblib')
        print(f"{name} trained successfully!", flush=True)

if __name__ == '__main__':
    main(x_train,y_train, MODELS)