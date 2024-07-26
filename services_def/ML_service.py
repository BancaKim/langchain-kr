# app/services/prediction_service.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = None
scaler = None

def train_model():
    global model, scaler

    file_path = r'C:\01DevelopKits\FinalProject\exel\fssDown\aa_fs2022_fs2023_1526.csv'
    df = pd.read_csv(file_path)
    df = df.dropna()

    X = df[['loan', 'IR',
            'asset2023', 'debt2023', 'equity2023', 'revenue2023', 'operatingincome2023', 'EBT2023', 'margin2023', 'turnover2023', 'leverage2023',
            'asset2022', 'debt2022', 'equity2022', 'revenue2022', 'operatingincome2022', 'EBT2022', 'margin2022', 'turnover2022', 'leverage2022',
            'asset2021', 'debt2021', 'equity2021', 'revenue2021', 'operatingincome2021', 'EBT2021', 'margin2021', 'turnover2021', 'leverage2021']]
    y = df['rate']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def preprocess_and_predict_proba(new_data):
    global model, scaler
    new_data_scaled = scaler.transform(new_data)
    probabilities = model.predict_proba(new_data_scaled)
    return probabilities