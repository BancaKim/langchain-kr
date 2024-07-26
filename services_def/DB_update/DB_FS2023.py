from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split

app = FastAPI()

# 학습된 모델과 스케일러를 전역 변수로 설정
model = None
scaler = None

class DataModel(BaseModel):
    loan: float
    IR: float
    asset2023: float
    debt2023: float
    equity2023: float
    revenue2023: float
    operatingincome2023: float
    EBT2023: float
    margin2023: float
    turnover2023: float
    leverage2023: float
    asset2022: float
    debt2022: float
    equity2022: float
    revenue2022: float
    operatingincome2022: float
    EBT2022: float
    margin2022: float
    turnover2022: float
    leverage2022: float
    asset2021: float
    debt2021: float
    equity2021: float
    revenue2021: float
    operatingincome2021: float
    EBT2021: float
    margin2021: float
    turnover2021: float
    leverage2021: float

def preprocess_and_predict_proba(new_data):
    global model, scaler
    
    # 새로운 데이터 전처리
    new_data_scaled = scaler.transform(new_data)
    
    # 확률 예측 수행
    probabilities = model.predict_proba(new_data_scaled)
    
    return probabilities

def train_model():
    global model, scaler

    # CSV 파일 경로 지정
    file_path = r'C:\01DevelopKits\FinalProject\exel\fssDown\aa_fs2022_fs2023_1526.csv'

    # CSV 파일 읽기
    df = pd.read_csv(file_path)

    # 결측값 처리 (예시로 결측값이 있는 행을 제거)
    df = df.dropna()

    # 특성과 목표 변수 설정 (열 이름은 실제 데이터에 맞게 변경)
    X = df[['loan', 'IR',
            'asset2023', 'debt2023', 'equity2023', 'revenue2023', 'operatingincome2023', 'EBT2023', 'margin2023', 'turnover2023', 'leverage2023',
            'asset2022', 'debt2022', 'equity2022', 'revenue2022', 'operatingincome2022', 'EBT2022', 'margin2022', 'turnover2022', 'leverage2022',
            'asset2021', 'debt2021', 'equity2021', 'revenue2021', 'operatingincome2021', 'EBT2021', 'margin2021', 'turnover2021', 'leverage2021']]  # 실제 데이터셋의 특성 열 이름으로 변경
    y = df['rate']  # 실제 데이터셋의 목표 변수 열 이름으로 변경

    # 데이터 스케일링 (선택사항)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE 적용
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 초기화
    model = RandomForestClassifier(n_estimators=150, random_state=42)

    # 모델 훈련
    model.fit(X_train, y_train)

    # 예측 수행
    y_pred = model.predict(X_test)

    # 모델 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

@app.on_event("startup")
def startup_event():
    train_model()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
