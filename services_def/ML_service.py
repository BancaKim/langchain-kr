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
import time
import pickle
import json
from io import BytesIO
import pandas as pd
from typing import Dict, List
from models.ML_model import ModelInfo
from mysqlConnector import create_connection, close_connection, Error
import logging

logger = logging.getLogger(__name__)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

def train_model():
    # file_path = r'C:\01DevelopKits\FinalProject\exel\fssDown\aa_fs2022_fs2023_1526.csv'
    file_path = r'C:\01DevelopKits\FinalProject\exel\fssDown\aa_fs2022_fs2023_07310945.csv'
    # 모델로딩이 너무 느려서 파일 가벼운 aa_fs2022_fs2023_07311059.csv 로 임시로 돌림
    df_origin = pd.read_csv(file_path)
    df = df_origin.dropna()

    feature_columns = [
            'asset2023', 'debt2023', 'equity2023', 'revenue2023', 'operatingincome2023', 'EBT2023', 'margin2023', 'turnover2023', 'leverage2023',
            'asset2022', 'debt2022', 'equity2022', 'revenue2022', 'operatingincome2022', 'EBT2022', 'margin2022', 'turnover2022', 'leverage2022',
            'asset2021', 'debt2021', 'equity2021', 'revenue2021', 'operatingincome2021', 'EBT2021', 'margin2021', 'turnover2021', 'leverage2021']

    X = df[feature_columns]
    y = df['rate']

    # 데이터 스케일링 (선택사항)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    
    # 데이터 불균형 확인
    logger.debug(file_path)
    logger.debug("Original dataset shape %s" % Counter(y))


    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # 랜덤포레스트 모델 초기화
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    # 모델 훈련
    
    
    logger.debug(" Credit Rating Model Learning start")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"Credit Rating Model Learning success in {elapsed_time:.2f} seconds")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)
    
    
        # Confusion Matrix 생성
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-'])

    model_info = {
        "model_name": "RandomForestClassifier",
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_estimators": model.n_estimators,
        "max_features": model.max_features,
        "feature_importances": model.feature_importances_.tolist(),
        "feature_names": model.feature_names_in_.tolist(),  # feature_names 추가
        "n_samples": len(df)
    }

    return model, scaler, accuracy, class_report, conf_matrix, model_info, df_origin, feature_columns



def save_model_info_to_db(model_info: Dict, accuracy: float, model, feature_columns: List[str]):
    try:
        # 모델 객체를 직렬화
        logger.debug('모델 객체를 직렬화 시작')
        model_binary = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        model_binary_size = len(model_binary)
        logger.debug(f'모델 객체를 직렬화 완료, 크기: {model_binary_size} 바이트')

        # 피처 컬럼 이름을 JSON으로 직렬화
        logger.debug('피처 컬럼 이름을 JSON으로 직렬화 시작')
        feature_columns_json = json.dumps(feature_columns)
        logger.debug('피처 컬럼 이름을 JSON으로 직렬화 완료')

        connection = create_connection()
        if connection:
            try:
                cursor = connection.cursor()
                logger.debug("데이터베이스에 연결됨")
                query = """
                INSERT INTO model_storage (model_name, creation_date, n_estimators, max_features, accuracy, feature_importances, n_samples, model_binary, feature_columns)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                data = (
                    model_info["model_name"],
                    model_info["creation_date"],
                    model_info["n_estimators"],
                    model_info["max_features"],
                    accuracy,
                    json.dumps(dict(zip(model_info["feature_names"], model_info["feature_importances"]))),
                    model_info["n_samples"],
                    model_binary,
                    feature_columns_json
                )
                cursor.execute(query, data)
                connection.commit()
                logger.info("모델이 저장되었습니다")
            except Error as e:
                logger.error(f"DB 삽입 중 오류 발생: {e}")
                connection.rollback()
            finally:
                close_connection(connection)
    except Exception as e:
        logger.error(f"모델 직렬화 중 오류 발생: {e}")
        raise e



# sql alchemy 방식
# def save_model_info_to_db(db: Session, model_info: Dict, accuracy: float, model, feature_columns: List[str]):
#     try:
#         # 모델 객체를 직렬화
#         logger.debug('모델 객체를 직렬화 시작')
#         model_binary = pickle.dumps(model)
#         logger.debug('모델 객체를 직렬화 완료')
        
#         logger.debug('피처 컬럼 이름을 JSON으로 직렬화 시작')
#         # 피처 컬럼 이름을 JSON으로 직렬화
#         feature_columns_json = json.dumps(feature_columns)
#         logger.debug('피처 컬럼 이름을 JSON으로 직렬화 완료')

#         logger.debug('모델 정보를 데이터베이스에 저장 시작')
#         # 모델 정보를 데이터베이스에 저장
#         model_record = ModelInfo(
#             model_name=model_info["model_name"],
#             creation_date=model_info["creation_date"],
#             n_estimators=model_info["n_estimators"],
#             max_features=model_info["max_features"],
#             accuracy=accuracy,
#             feature_importances=json.dumps(dict(zip(model_info["feature_names"], model_info["feature_importances"]))),
#             n_samples=model_info["n_samples"],
          
#             feature_columns=feature_columns_json,
#         )
        
#         logger.debug(f'model_info: {model_info}')
#         logger.debug(f'accuracy: {accuracy}')
#         logger.debug(f'model: {model}')
#         logger.debug(f'feature_columns: {feature_columns}')
        
#         logger.debug('db.add(model_record) 직전')
#         db.add(model_record)
#         logger.debug('db.add(model_record) 완료')
#         db.commit()
#         logger.debug('db.commit() 완료')
#         db.refresh(model_record)
#         logger.debug('db.refresh(model_record) 완료')
#         return model_record
#     except Exception as e:
#         logger.error(f"DB 삽입 중 오류 발생: {e}")
#         db.rollback()
#         raise e



    
def load_model_info_from_db(db: Session, model_id: int):
    model_record = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()
    if model_record:
        model = pickle.loads(model_record.model_binary)  # 직렬화된 모델 객체를 역직렬화
        feature_columns = json.loads(model_record.feature_columns)  # 피처 컬럼 이름을 JSON에서 복원

        # df_origin을 CSV에서 복원
        df_origin_csv = BytesIO(model_record.df_origin)
        df_origin = pd.read_csv(df_origin_csv)

        return model, feature_columns, df_origin
    else:
        return None, None, None








# generate_predictions 에서 사용 실제로 신용등급 산출하는 로직
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

## 등급 부여를 위한 법인 정보 추출
        # "SELECT a.corp_name, a.jurir_no, COALESCE(g.IR등급, 7) AS IR, "
        # "b.totalAsset2023 AS asset2023, b.totalDebt2023 AS debt2023, b.totalEquity2023 AS equity2023, "
        # "b.revenue2023 AS revenue2023, b.operatingIncome2023 AS operatingincome2023, b.earningBeforeTax2023 AS EBT2023, "
        # "b.margin2023 AS margin2023, b.turnover2023 AS turnover2023, b.leverage2023 AS leverage2023, "
        # "c.totalAsset2022 AS asset2022, c.totalDebt2022 AS debt2022, c.totalEquity2022 AS equity2022, "
        # "c.revenue2022 AS revenue2022, c.operatingIncome2022 AS operatingincome2022, c.earningBeforeTax2022 AS EBT2022, "
        # "c.margin2022 AS margin2022, c.turnover2022 AS turnover2022, c.leverage2022 AS leverage2022, "
        # "d.totalAsset2021 AS asset2021, d.totalDebt2021 AS debt2021, d.totalEquity2021 AS equity2021, "
        # "d.revenue2021 AS revenue2021, d.operatingIncome2021 AS operatingincome2021, d.earningBeforeTax2021 AS EBT2021, "
        # "d.margin2021 AS margin2021, d.turnover2021 AS turnover2021, d.leverage2021 AS leverage2021 "
        # "FROM companyInfo a "
        # "LEFT OUTER JOIN FS2023 b ON a.jurir_no = b.jurir_no "
        # "LEFT OUTER JOIN FS2022 c ON a.jurir_no = c.jurir_no "
        # "LEFT OUTER JOIN FS2021 d ON a.jurir_no = d.jurir_no "
        # "LEFT OUTER JOIN kb_data_v1_copy e ON TRIM(e.법인번호) = a.jurir_no "
        # "LEFT OUTER JOIN IRrate g ON e.산업분류코드 = g.표준산업분류 "
        # "WHERE a.jurir_no IN :jurir_no_list"
        
# 예측을 위한 데이터 추출
def get_new_data_from_db(db: Session, jurir_no_list: list):
    query = text(
        "SELECT a.corp_name, a.jurir_no,"
        "b.totalAsset2023 AS asset2023, b.totalDebt2023 AS debt2023, b.totalEquity2023 AS equity2023, "
        "b.revenue2023 AS revenue2023, b.operatingIncome2023 AS operatingincome2023, b.earningBeforeTax2023 AS EBT2023, "
        "b.margin2023 AS margin2023, b.turnover2023 AS turnover2023, b.leverage2023 AS leverage2023, "
        "c.totalAsset2022 AS asset2022, c.totalDebt2022 AS debt2022, c.totalEquity2022 AS equity2022, "
        "c.revenue2022 AS revenue2022, c.operatingIncome2022 AS operatingincome2022, c.earningBeforeTax2022 AS EBT2022, "
        "c.margin2022 AS margin2022, c.turnover2022 AS turnover2022, c.leverage2022 AS leverage2022, "
        "d.totalAsset2021 AS asset2021, d.totalDebt2021 AS debt2021, d.totalEquity2021 AS equity2021, "
        "d.revenue2021 AS revenue2021, d.operatingIncome2021 AS operatingincome2021, d.earningBeforeTax2021 AS EBT2021, "
        "d.margin2021 AS margin2021, d.turnover2021 AS turnover2021, d.leverage2021 AS leverage2021 "
        "FROM FS2023 b "
        "LEFT OUTER JOIN companyInfo a ON a.jurir_no = b.jurir_no "
        "LEFT OUTER JOIN FS2022 c ON a.jurir_no = c.jurir_no "
        "LEFT OUTER JOIN FS2021 d ON a.jurir_no = d.jurir_no "
        "WHERE b.totalAsset2023 > 0 limit 100;"
    )

    result = db.execute(query)
    data = result.fetchall()
    columns = result.keys()
    new_data = pd.DataFrame(data, columns=columns)
    
    # logger.debug(new_data)
    
    return new_data


# generate_predictions에서 생성된 예상치를 DB에 삽입
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

# 여기서 법인 리스트 숫자 제한
def get_jurir_no_list(db: Session):
    query = text("""
    select jurir_no from FS2023 where totalAsset2023 > 0 limit 100;
    """)
    result = db.execute(query)
    jurir_no_list = [row[0] for row in result.fetchall()]
    logger.debug(f"Size of jurir_no_list: {len(jurir_no_list)}")
    return jurir_no_list

# 재무정보 존재하는 법인번호 리스트 받아서 신용등급 산출 insert_predictions에서 사용
def generate_predictions(db: Session, model, scaler):
    jurir_no_list = get_jurir_no_list(db)
    logger.debug(f"Size of jurir_no_list: {len(jurir_no_list)}")
    if not jurir_no_list:
        return {"error": "No data found in database for prediction"}, []

    new_data = get_new_data_from_db(db, jurir_no_list)
    if new_data.empty:
        return {"error": "Data not found for the given jurir_no list"}, []
    
        # 중복된 jurir_no 제거
    new_data = new_data.drop_duplicates(subset=['jurir_no'])

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
    # logger.debug(f"Size of predictions: {len(predictions)}")
    
    # logger.debug(predictions)

    return None, predictions

# predict all에서 사용
def generate_predictions_dictionary(db: Session, model, scaler):
    jurir_no_list = get_jurir_no_list(db)
    new_data = get_new_data_from_db(db, jurir_no_list)
    if new_data.empty:
        return {"error": "Data not found for the given jurir_no list"}, []
            # 중복된 jurir_no 제거
    # new_data = new_data.drop_duplicates(subset=['jurir_no'])

    predictions = []
    for _, row in new_data.iterrows():
        data = row.to_frame().T
        probabilities = preprocess_and_predict_proba(data, model, scaler)
        class_probabilities = list(zip(model.classes_, probabilities[0]))
        class_probabilities.sort(key=lambda x: x[1], reverse=True)

        sorted_probabilities = {}
        for cls, prob in class_probabilities:
            sorted_probabilities[cls] = round(prob, 2)

        result = {
            "jurir_no": row["jurir_no"],
            "corp_name": row["corp_name"],
            "sorted_probabilities": sorted_probabilities
        }

        predictions.append(result)
    
    
    logger.debug(f"Size of jurir_no_list: {len(jurir_no_list)}")
    logger.debug(f"Size of predictions: {len(predictions)}")
    return None, predictions


# view_DB_predict 에서 사용
def get_db_predictions(db: Session):
    query = text("""
    SELECT *
    FROM predict_ratings pr
    WHERE pr.corporate_number IN (
    SELECT jurir_no 
    FROM (
    SELECT DISTINCT a.jurir_no, b.totalAsset2023
    FROM companyInfo a
    LEFT JOIN FS2023 b ON a.jurir_no = b.jurir_no
    WHERE b.totalAsset2023 > 0
    ORDER BY b.totalAsset2023 DESC

    ) AS subquery
    )
    AND model_reference = (
        SELECT model_reference
        FROM spoon.predict_ratings
        WHERE base_year = '2023'
        ORDER BY timestamp DESC
        LIMIT 1
    )

    """)
    result = db.execute(query)
    predictions = result.fetchall()
    columns = result.keys()
    predictions_dict = [dict(zip(columns, row)) for row in predictions]
    
    return predictions_dict