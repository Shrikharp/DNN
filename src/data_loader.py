import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    # Fill missing values with column mean
    df = df.fillna(df.mean(numeric_only=True))

    # Convert target
    df['CDR'] = df['CDR'].apply(lambda x: 0 if x == 0 else 1)

    features = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    X = df[features].values
    y = df['CDR'].values

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y