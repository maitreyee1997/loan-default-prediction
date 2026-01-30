import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    # Handle missing values
    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
    df['loan_int_rate'].fillna(df['loan_int_rate'].mean(), inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    df['loan_grade'] = le.fit_transform(df['loan_grade'])
    df['cb_person_default_on_file'] = le.fit_transform(df['cb_person_default_on_file'])

    df = pd.get_dummies(
        df,
        columns=['person_home_ownership', 'loan_intent'],
        drop_first=True
    )

    # Split features & target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
