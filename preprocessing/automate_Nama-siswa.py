import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib

# Fungsi preprocessing lengkap
def preprocessing_pipeline(csv_path):
    df = pd.read_csv(csv_path)

    # 1. Drop fitur yang tidak digunakan
    df = df.drop(columns=['Customer_ID', 'Month', 'Occupation', 'Type_of_Loan', 'Credit_Utilization_Ratio'], axis=1)

    # 2. Split data menjadi train dan test
    train_df, test_df = train_test_split(df, test_size=0.05, random_state=42, shuffle=True)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # 3. Undersampling untuk data imbalance
    df_majority_1 = train_df[(train_df.Credit_Score == "Standard")]
    df_majority_2 = train_df[(train_df.Credit_Score == "Poor")]
    df_minority = train_df[(train_df.Credit_Score == "Good")]

    df_majority_1_us = resample(df_majority_1, n_samples=16936, random_state=42)
    df_majority_2_us = resample(df_majority_2, n_samples=16936, random_state=42)

    undersampled_train_df = pd.concat([df_minority, df_majority_1_us, df_majority_2_us]).reset_index(drop=True)
    undersampled_train_df = shuffle(undersampled_train_df, random_state=42)

    X_train = undersampled_train_df.drop(columns="Credit_Score", axis=1)
    y_train = undersampled_train_df["Credit_Score"]

    X_test = test_df.drop(columns="Credit_Score", axis=1)
    y_test = test_df["Credit_Score"]

    # 4. Scaling data numerikal
    numerical_columns = [
        'Age', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly',
        'Monthly_Balance', 'Credit_History_Age'
    ]

    def scaling(features, df, df_test):
        for feature in features:
            scaler = MinMaxScaler()
            scaler.fit(df[[feature]])
            df[feature] = scaler.transform(df[[feature]])
            df_test[feature] = scaler.transform(df_test[[feature]])
            joblib.dump(scaler, f"preprocessing/model/scaler_{feature}.joblib")
        return df, df_test

    X_train, X_test = scaling(numerical_columns, X_train, X_test)

    # 5. Encoding fitur kategorikal
    categorical_columns = ['Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

    def encoding(features, df, df_test):
        for feature in features:
            encoder = LabelEncoder()
            encoder.fit(df[feature])
            df[feature] = encoder.transform(df[feature])
            df_test[feature] = encoder.transform(df_test[feature])
            joblib.dump(encoder, f"preprocessing/model/encoder_{feature}.joblib")
        return df, df_test

    X_train, X_test = encoding(categorical_columns, X_train, X_test)

    # Encode target
    target_encoder = LabelEncoder()
    target_encoder.fit(y_train)
    y_train_enc = target_encoder.transform(y_train)
    y_test_enc = target_encoder.transform(y_test)
    joblib.dump(target_encoder, "preprocessing/model/encoder_target.joblib")

    # 6. PCA Reduction
    pca_numerical_columns_1 = [
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
        'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_History_Age'
    ]

    pca_numerical_columns_2 = [
        'Monthly_Inhand_Salary', 'Monthly_Balance', 'Amount_invested_monthly', 'Total_EMI_per_month'
    ]

    # PCA pertama
    pca_1 = PCA(n_components=5, random_state=123)
    pca_1.fit(X_train[pca_numerical_columns_1])
    joblib.dump(pca_1, "preprocessing/model/pca_1.joblib")
    X_train_pca_1 = pca_1.transform(X_train[pca_numerical_columns_1])
    X_test_pca_1 = pca_1.transform(X_test[pca_numerical_columns_1])

    for i in range(5):
        X_train[f'pc1_{i+1}'] = X_train_pca_1[:, i]
        X_test[f'pc1_{i+1}'] = X_test_pca_1[:, i]

    X_train.drop(columns=pca_numerical_columns_1, inplace=True)
    X_test.drop(columns=pca_numerical_columns_1, inplace=True)

    # PCA kedua
    pca_2 = PCA(n_components=2, random_state=123)
    pca_2.fit(X_train[pca_numerical_columns_2])
    joblib.dump(pca_2, "preprocessing/model/pca_2.joblib")
    X_train_pca_2 = pca_2.transform(X_train[pca_numerical_columns_2])
    X_test_pca_2 = pca_2.transform(X_test[pca_numerical_columns_2])

    for i in range(2):
        X_train[f'pc2_{i+1}'] = X_train_pca_2[:, i]
        X_test[f'pc2_{i+1}'] = X_test_pca_2[:, i]

    X_train.drop(columns=pca_numerical_columns_2, inplace=True)
    X_test.drop(columns=pca_numerical_columns_2, inplace=True)

    # 7. Simpan hasil
    train_final = pd.concat([X_train, pd.Series(y_train_enc, name='Credit_Score')], axis=1)
    test_final = pd.concat([X_test, pd.Series(y_test_enc, name='Credit_Score')], axis=1)

    return train_final, test_final
file_path = f'./train_cleaned.csv'
train_final, test_final = preprocessing_pipeline(file_path)
train_final.to_csv("train_pca.csv", index=False)
test_final.to_csv("test_pca.csv", index=False)