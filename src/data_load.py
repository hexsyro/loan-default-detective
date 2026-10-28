import pandas as pd


def create_df(path):
    #simplying data input
    cols = ['title', 'loan_amnt', 'int_rate','dti', 'revol_util','open_acc','installment', 'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'loan_status']
    df=pd.read_csv(path, usecols=cols)

    df=df.rename(columns={'title':'purpose'})

    return df

def emp_length_col(df):

    order = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
        "10+ years": 10,
    }

    df["emp_length"] = df["emp_length"].map(order)

    text_cols =df.select_dtypes(include=["object", "str"]).columns.tolist()

    return df,text_cols

def defaulted_col(df):
    df = df[df["loan_status"].isin([
        "Fully Paid",
        "Charged Off",
        "Does not meet the credit policy. Status:Charged Off",
        "Default"
    ])]

    df['defaulted']=(df['loan_status'].isin(['Default',
                                        'Does not meet the credit policy. Status:Charged Off',
                                        'Charged Off'])).astype(int)
    return df

def main():
    #All Lending Club loan data - https://www.kaggle.com/datasets/wordsforthewise/lending-club
    path='data/raw/raw_loan_data.csv'
    df=create_df(path)
    df, text_cols=emp_length_col(df)
    print(text_cols)
    df=defaulted_col(df)
    df.info()
    df.head()
    df.describe()
    
    df.to_csv('data/processed/loans_cleaned.csv', index=False)

    print(df['defaulted'].value_counts(normalize=True))


if __name__ == '__main__':
    main()