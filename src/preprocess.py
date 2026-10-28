import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder



def fix_purpose(df):
    df['purpose'] = df["purpose"].str.lower().str.strip()

    counts = df.loc[df["defaulted"] == 0, "purpose"].value_counts()
    keep = counts[counts >= 500].index

    df = df[df["purpose"].isin(keep)]
    return df


def encode(df):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None)     # unseen categories -> all zeros
    enc = pd.DataFrame(ohe.fit_transform(df[["home_ownership","purpose"]]), columns=ohe.get_feature_names_out())
    df = pd.concat([df.drop(columns=["home_ownership","purpose"]), enc], axis=1)
    return df

def split(df):
    df=df.dropna()  # drop rows with missing values
    X=df.drop(columns=["loan_status","emp_title", "defaulted"])
    Y=df["defaulted"].astype(int)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y,           # stratify keeps class ratios equal (classification)
        random_state=42)
    return X_train, X_test, Y_train, Y_test

def fix_income(X_train, X_test):
    lo, hi = X_train["annual_inc"].quantile([0.001, 0.999])
    X_train["annual_inc"] = X_train["annual_inc"].clip(lo, hi)
    X_test["annual_inc"] = X_test["annual_inc"].clip(lo, hi)
    return X_train, X_test

def scale(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test

def export(scaled_X_train, scaled_X_test, X_train, Y_train, X_test, Y_test):
    scaled_X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns, index=X_train.index)
    scaled_X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns, index=X_test.index)
    X_train = pd.DataFrame(X_train, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test, columns=X_test.columns, index=X_test.index)
    Y_train = pd.DataFrame(Y_train, columns=['defaulted'], index=Y_train.index)
    Y_test = pd.DataFrame(Y_test, columns=['defaulted'], index=Y_test.index)

    scaled_X_train.to_csv('data/processed/X_train_scaled.csv', index=False)
    scaled_X_test.to_csv('data/processed/X_test_scaled.csv', index=False)
    Y_train.to_csv('data/processed/Y_train.csv', index=False)
    Y_test.to_csv('data/processed/Y_test.csv', index=False)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)


def main():
    df=pd.read_csv('data/processed/loans_cleaned.csv')

    # fix column data
    df = fix_purpose(df)
    print()
    print(f"there is {df["purpose"].nunique()} amount of loan purposes in this dataset.")

    #one hot encode
    df = encode(df)
    print()
    print(df.columns)
    df = df.dropna(subset='int_rate')
    print(df['defaulted'].value_counts(normalize=True))


    #split dataset into test and train
    X_train, X_test, Y_train, Y_test=split(df)    
    print()
    print(X_train.shape, X_test.shape, Y_train.mean().round(3), Y_test.mean().round(3))

    X_train, X_test=fix_income(X_train, X_test)

    scaled_X_train, scaled_X_test=scale(X_train, X_test)

    export(scaled_X_train, scaled_X_test, X_train, Y_train, X_test, Y_test)
    print()

    #check the output files
    check_Scaled_X = pd.read_csv("data/processed/X_train_scaled.csv")
    check_X = pd.read_csv("data/processed/X_train.csv")
    check_Y = pd.read_csv("data/processed/Y_train.csv")
    print(check_X.shape, check_Scaled_X.shape, check_Y.shape)
    print(len(check_X) == len(check_Scaled_X) == len(check_Y))  # True


if __name__ == '__main__':
    main()