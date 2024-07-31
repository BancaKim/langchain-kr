import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text, bindparam
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from datetime import datetime
import json
from sqlalchemy.orm import Session

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

def train_model():
    file_path = r'C:\01DevelopKits\FinalProject\exel\fssDown\aa_fs2022_fs2023_1526.csv'
    df = pd.read_csv(file_path)
    df = df.dropna()

    feature_columns = ['IR',
            'asset2023', 'debt2023', 'equity2023', 'revenue2023', 'operatingincome2023', 'EBT2023', 'margin2023', 'turnover2023', 'leverage2023',
            'asset2022', 'debt2022', 'equity2022', 'revenue2022', 'operatingincome2022', 'EBT2022', 'margin2022', 'turnover2022', 'leverage2022',
            'asset2021', 'debt2021', 'equity2021', 'revenue2021', 'operatingincome2021', 'EBT2021', 'margin2021', 'turnover2021', 'leverage2021']

    X = df[feature_columns]
    y = df['rate']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    model_info = {
        "model_name": "RandomForestClassifier",
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_estimators": model.n_estimators,
        "max_features": model.max_features,
        "feature_importances": model.feature_importances_.tolist(),
        "n_samples": len(df)
    }

    return model, scaler, accuracy, class_report, conf_matrix, model_info

def preprocess_and_predict_proba(new_data, model, scaler):
    if model is None:
        raise ValueError("The model is not trained yet. Please train the model before prediction.")
    if scaler is None:
        raise ValueError("The scaler is not initialized. Please initialize the scaler before prediction.")
    
    # 모델이 학습할 때 사용된 피처 이름을 가져옵니다
    feature_names = model.feature_names_in_
    
    # 'jurir_no' 열을 제거하고 피처 이름에 맞게 정렬
    new_data = new_data.drop(columns=['jurir_no'])
    new_data = new_data.drop(columns=['corp_name'])
    new_data = new_data[feature_names]
    
    new_data_scaled = scaler.transform(new_data)
    probabilities = model.predict_proba(new_data_scaled)
    return probabilities


def get_new_data_from_db(db: Session, jurir_no_list: list):
    query = text(
        "SELECT a.corp_name, a.jurir_no, COALESCE(g.IR등급, 7) AS IR, "
        "b.totalAsset2023 AS asset2023, b.totalDebt2023 AS debt2023, b.totalEquity2023 AS equity2023, "
        "b.revenue2023 AS revenue2023, b.operatingIncome2023 AS operatingincome2023, b.earningBeforeTax2023 AS EBT2023, "
        "b.margin2023 AS margin2023, b.turnover2023 AS turnover2023, b.leverage2023 AS leverage2023, "
        "c.totalAsset2022 AS asset2022, c.totalDebt2022 AS debt2022, c.totalEquity2022 AS equity2022, "
        "c.revenue2022 AS revenue2022, c.operatingIncome2022 AS operatingincome2022, c.earningBeforeTax2022 AS EBT2022, "
        "c.margin2022 AS margin2022, c.turnover2022 AS turnover2022, c.leverage2022 AS leverage2022, "
        "d.totalAsset2021 AS asset2021, d.totalDebt2021 AS debt2021, d.totalEquity2021 AS equity2021, "
        "d.revenue2021 AS revenue2021, d.operatingIncome2021 AS operatingincome2021, d.earningBeforeTax2021 AS EBT2021, "
        "d.margin2021 AS margin2021, d.turnover2021 AS turnover2021, d.leverage2021 AS leverage2021 "
        "FROM companyInfo a "
        "LEFT OUTER JOIN FS2023 b ON a.jurir_no = b.jurir_no "
        "LEFT OUTER JOIN FS2022 c ON a.jurir_no = c.jurir_no "
        "LEFT OUTER JOIN FS2021 d ON a.jurir_no = d.jurir_no "
        "LEFT OUTER JOIN kb_data_v1_copy e ON TRIM(e.법인번호) = a.jurir_no "
        "LEFT OUTER JOIN IRrate g ON e.산업분류코드 = g.표준산업분류 "
        "WHERE a.jurir_no IN :jurir_no_list"
    ).bindparams(bindparam('jurir_no_list', expanding=True))

    result = db.execute(query, {"jurir_no_list": jurir_no_list})
    data = result.fetchall()
    columns = result.keys()
    new_data = pd.DataFrame(data, columns=columns)
    
    logger.debug(new_data)
    
    return new_data


def insert_predictions_into_db(db: Session, prediction: dict, model_reference: str):
    query = text("""
    INSERT INTO predict_ratings (
        corporate_number, company_name, base_year, AAA_plus, AAA, AAA_minus,
        AA_plus, AA, AA_minus, A_plus, A, A_minus, BBB_plus, BBB, BBB_minus,
        BB_plus, BB, BB_minus, B_plus, B, B_minus, CCC_plus, CCC, CCC_minus, C, D, model_reference
    ) VALUES (
        :jurir_no, :corp_name, :base_year, :AAA_plus, :AAA, :AAA_minus, :AA_plus, :AA, :AA_minus,
        :A_plus, :A, :A_minus, :BBB_plus, :BBB, :BBB_minus, :BB_plus, :BB, :BB_minus, :B_plus, :B,
        :B_minus, :CCC_plus, :CCC, :CCC_minus, :C, :D, :model_reference
    )
    """)
    db.execute(query, {
        "jurir_no": prediction["jurir_no"],
        "corp_name": prediction["corp_name"],
        "base_year": prediction["base_year"],
        "AAA_plus": prediction["AAA_plus"],
        "AAA": prediction["AAA"],
        "AAA_minus": prediction["AAA_minus"],
        "AA_plus": prediction["AA_plus"],
        "AA": prediction["AA"],
        "AA_minus": prediction["AA_minus"],
        "A_plus": prediction["A_plus"],
        "A": prediction["A"],
        "A_minus": prediction["A_minus"],
        "BBB_plus": prediction["BBB_plus"],
        "BBB": prediction["BBB"],
        "BBB_minus": prediction["BBB_minus"],
        "BB_plus": prediction["BB_plus"],
        "BB": prediction["BB"],
        "BB_minus": prediction["BB_minus"],
        "B_plus": prediction["B_plus"],
        "B": prediction["B"],
        "B_minus": prediction["B_minus"],
        "CCC_plus": prediction["CCC_plus"],
        "CCC": prediction["CCC"],
        "CCC_minus": prediction["CCC_minus"],
        "C": prediction["C"],
        "D": prediction["D"],
        "model_reference": model_reference
    })
    db.commit()
    
    

def generate_predictions(db: Session, model, scaler, jurir_no_list: list):
    new_data = get_new_data_from_db(db, jurir_no_list)
    if new_data.empty:
        return {"error": "Data not found for the given jurir_no list"}, []

    predictions = []
    for _, row in new_data.iterrows():
        data = row.to_frame().T
        probabilities = preprocess_and_predict_proba(data, model, scaler)
        class_probabilities = list(zip(model.classes_, probabilities[0]))
        class_probabilities.sort(key=lambda x: x[1], reverse=True)
        
        sorted_probabilities = []
        for cls, prob in class_probabilities:
            sorted_probabilities.append({
                'class': cls, 
                'probability': round(prob, 2)
            })
        
        result = {
            "jurir_no": row["jurir_no"],
            "corp_name": row["corp_name"],
            "sorted_probabilities": sorted_probabilities
        }
        
        predictions.append(result)

    return None, predictions