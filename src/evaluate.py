from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

import pandas as pd
import numpy as np
import joblib


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluatr_classier(model, X_test, Y_test):
    preds=model.predict(X_test)
    proba_preds=get_scores(model,X_test)
    
    accuracy=accuracy_score(Y_test,preds)
    precision=precision_score(Y_test,preds, zero_division=0)
    recall=recall_score(Y_test,preds)
    f1=f1_score(Y_test,preds)
    roc_auc=roc_auc_score(Y_test,proba_preds)

    return {
        "Accuracy" :accuracy, 
        "Precision" :precision,
        "Recall" :recall,
        "F1" :f1,
        "ROC AUC":roc_auc,}


def main():
    results={}
    x_test=pd.read_csv('data/processed/x_test_scaled.csv')
    y_test=pd.read_csv("data/processed/Y_test.csv").squeeze()
    print("Here is the evaluation metrics scores for 6 different ML classifiers.")
    print("always-0 accuracy:", round(1 - y_test.mean(), 3))
    print()
    MD_names = {
        "LR": "Logistic Reg",
        "SVM": "Support Vector",
        "KNN": "K-Neighbors",
        "GNB": "Naive Bayes",
        "RFC": "Random Forest",
        "DTC": "Decision Tree",
    }

    for name in ["LR", "SVM", "KNN", "GNB", "RFC", "DTC"]:
        model = joblib.load(f"models/{name}.joblib")
        results[name]=evaluatr_classier(model, x_test, y_test)
        print(f"{MD_names[name]} classifier for predicting default in a loan.")
        print(evaluatr_classier(model, x_test, y_test))
        print()
    print()
    print("===========================================================================================")
    table = pd.DataFrame(results).T.round(3)
    table = table.sort_values("ROC AUC", ascending=False)
    print(table)
    print()
    table.to_csv('outputs/model_scores.csv')
    print("model evaluation table is saved into output/model_scores.csv")


if __name__ == '__main__':
    main()