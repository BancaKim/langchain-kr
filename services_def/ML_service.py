import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text, desc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sqlalchemy.exc import SQLAlchemyError
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
import numpy as np  # numpy를 import 해야 합니다
from typing import Union
import logging
import os
import joblib
import uuid
from models.ML_model import ModelStorage


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

    # model_info = {
    #     "model_name": "RandomForestClassifier",
    #     "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #     "n_estimators": model.n_estimators,
    #     "max_features": model.max_features,
    #     "feature_importances": model.feature_importances_.tolist(),
    #     "feature_names": model.feature_names_in_.tolist(),  # feature_names 추가
    #     "n_samples": len(df)
    # }
    
    model_info = {
    "model_name": "RandomForestClassifier",
    "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_estimators": model.n_estimators,
    "max_features": model.max_features,
    "feature_importances": dict(zip(model.feature_names_in_, model.feature_importances_)),  # 수정된 부분
    "feature_names": model.feature_names_in_.tolist(),
    "n_samples": len(df)
    }

    return model, scaler, accuracy, class_report, conf_matrix, model_info, df_origin, feature_columns



def get_model_info_by_id(db: Session, model_id: str):
    model_info = db.query(ModelStorage).filter(ModelStorage.model_id == model_id).first()

    if model_info:
        return {
            "model_id": model_info.model_id,
            "model_name": model_info.model_name,
            "creation_date": model_info.creation_date.strftime("%Y-%m-%d %H:%M:%S"),
            "n_estimators": model_info.n_estimators,
            "max_features": model_info.max_features,
            "accuracy": model_info.accuracy,
            "feature_importances": model_info.feature_importances,
            "n_samples": model_info.n_samples,
            "model_filepath": model_info.model_filepath,
            "feature_columns": model_info.feature_columns,
            "scaler": model_info.scaler,
            "class_report": model_info.class_report,
            "conf_matrix": model_info.conf_matrix
        }
    else:
        return None






def save_model_info_to_db(model_info: Dict, accuracy: float, model, scaler, class_report: Dict, conf_matrix: np.ndarray, feature_columns: List[str], db: Session):
    try:
        # 모델 저장 폴더 생성
        model_dir = 'saved_models'
        os.makedirs(model_dir, exist_ok=True)

        # 고유한 모델 ID 생성
        model_id = str(uuid.uuid4())

        # 모델 파일 경로 설정
        model_filename = f"{model_id}_{model_info['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model_filepath = os.path.join(model_dir, model_filename)

        # 모델 파일로 저장
        logger.debug('모델을 파일로 저장 시작')
        joblib.dump(model, model_filepath)
        logger.debug(f'모델을 파일로 저장 완료: {model_filepath}')

        # 피처 컬럼 이름을 JSON으로 직렬화
        logger.debug('피처 컬럼 이름을 JSON으로 직렬화 시작')
        feature_columns_json = json.dumps(feature_columns)
        logger.debug('피처 컬럼 이름을 JSON으로 직렬화 완료')

        # 기타 정보 직렬화
        class_report_json = json.dumps(class_report)
        conf_matrix_json = json.dumps(conf_matrix.tolist())  # ndarray를 리스트로 변환

        # SQLAlchemy를 사용하여 데이터베이스에 모델 정보 저장
        model_storage = ModelStorage(
            model_id=model_id,
            model_name=model_info["model_name"],
            creation_date=datetime.now(),
            n_estimators=model_info["n_estimators"],
            max_features=model_info["max_features"],
            accuracy=accuracy,
            feature_importances=model_info['feature_importances'],  # dict 형태 그대로 저장
            n_samples=model_info["n_samples"],
            model_filepath=model_filepath,
            feature_columns=feature_columns_json,
            scaler=pickle.dumps(scaler),  # Scaler 객체를 바이너리로 저장
            class_report=class_report_json,
            conf_matrix=conf_matrix_json  # 변환된 conf_matrix를 사용
        )

        db.add(model_storage)
        db.commit()
        logger.info(f"모델 정보가 데이터베이스에 저장되었습니다. 모델 ID: {model_id}")
    
    except Exception as e:
        db.rollback()
        logger.error(f"모델 파일 저장 중 오류 발생: {e}")
        raise e
    
    return model_id

    
    
    
    
def get_all_model_info(db: Session):
    try:
        # creation_date 기준으로 내림차순 정렬
        model_info_list = db.query(ModelStorage).order_by(desc(ModelStorage.creation_date)).all()

        result = []
        for model_info in model_info_list:
            result.append({
                "model_id": model_info.model_id,
                "model_name": model_info.model_name,
                "creation_date": model_info.creation_date.strftime("%Y-%m-%d %H:%M:%S"),
                "n_estimators": model_info.n_estimators,
                "max_features": model_info.max_features,
                "accuracy": model_info.accuracy,
                "feature_importances": model_info.feature_importances,
                "n_samples": model_info.n_samples,
                "model_filepath": model_info.model_filepath,
                "feature_columns": model_info.feature_columns,  # JSON 파싱하지 않음
                "scaler": model_info.scaler,
                "class_report": model_info.class_report,  # JSON 파싱하지 않음
                "conf_matrix": model_info.conf_matrix,  # JSON 파싱하지 않음
                "is_default": model_info.is_default,  # 디폴트 여부 포함
            })

        return result

    except Exception as e:
        logger.error(f"모델 정보를 가져오는 중 오류 발생: {e}")
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



    
# def load_model_info_from_db(db: Session, model_id: int):
#     model_record = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()
#     if model_record:
#         model = pickle.loads(model_record.model_binary)  # 직렬화된 모델 객체를 역직렬화
#         feature_columns = json.loads(model_record.feature_columns)  # 피처 컬럼 이름을 JSON에서 복원

#         # df_origin을 CSV에서 복원
#         df_origin_csv = BytesIO(model_record.df_origin)
#         df_origin = pd.read_csv(df_origin_csv)

#         return model, feature_columns, df_origin
#     else:
#         return None, None, None








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
    # jurir_no_list를 SQL IN 조건에 사용할 수 있는 문자열로 변환
    jurir_no_tuple = tuple(jurir_no_list)
    
    query = text(
        "SELECT a.corp_name, a.jurir_no,"
        "b.totalAsset2023, b.totalDebt2023, b.totalEquity2023, "
        "b.revenue2023, b.operatingIncome2023, b.earningBeforeTax2023, "
        "b.margin2023, b.turnover2023, b.leverage2023, "
        "c.totalAsset2022, c.totalDebt2022, c.totalEquity2022, "
        "c.revenue2022, c.operatingIncome2022, c.earningBeforeTax2022, "
        "c.margin2022, c.turnover2022, c.leverage2022, "
        "d.totalAsset2021, d.totalDebt2021, d.totalEquity2021, "
        "d.revenue2021, d.operatingIncome2021, d.earningBeforeTax2021, "
        "d.margin2021, d.turnover2021, d.leverage2021 "
        "FROM FS2023 b "
        "LEFT OUTER JOIN companyInfo a ON a.jurir_no = b.jurir_no "
        "LEFT OUTER JOIN FS2022 c ON a.jurir_no = c.jurir_no "
        "LEFT OUTER JOIN FS2021 d ON a.jurir_no = d.jurir_no "
        "WHERE b.totalAsset2023 > 0 "
        "AND a.corp_name NOT LIKE '%은행%' "
        "AND a.corp_name NOT LIKE '%금융지주%' "
        "AND a.corp_name NOT LIKE '%보험%' "
        "AND a.jurir_no IN :jurir_no_list "
        "ORDER BY b.totalAsset2023 DESC;"
    )
    
    # 쿼리 실행 시 파라미터로 jurir_no_list 전달
    result = db.execute(query, {'jurir_no_list': jurir_no_tuple})
    data = result.fetchall()
    columns = result.keys()
    new_data = pd.DataFrame(data, columns=columns)
    
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
    jurir_no_list = get_jurir_no_list(db) # select jurir_no from FS2023 where totalAsset2023 > 0 limit 100;
    
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

# predict all에서 사용 모든 기업의 등급을 산출하므로 수정필요. 
def generate_predictions_dictionary(db: Session, model, scaler, jurir_no=None):
    logger.info("Starting generate_predictions_dictionary function")
    logger.info(f"Received jurir_no: {jurir_no}")

    # 만약 jurir_no가 제공되지 않으면, 데이터베이스에서 jurir_no_list를 가져옵니다.
    if jurir_no:
        jurir_no_list = [jurir_no]
        logger.info(f"Using provided jurir_no: {jurir_no_list}")
    else:
        jurir_no_list = get_jurir_no_list(db)
        logger.info(f"Retrieved jurir_no_list from database: {jurir_no_list}")
    
    # 새로운 데이터를 데이터베이스에서 가져옵니다.
    new_data = get_new_data_from_db(db, jurir_no_list)
    if new_data.empty:
        logger.error("No data found for the provided jurir_no list")
        return {"error": "Data not found for the given jurir_no list"}, []

    logger.info(f"Retrieved new data from database. Number of records: {len(new_data)}")

    predictions = []
    for _, row in new_data.iterrows():
        logger.debug(f"Processing row for jurir_no: {row['jurir_no']}")
        
        data = row.to_frame().T
        probabilities = preprocess_and_predict_proba(data, model, scaler)
        logger.debug(f"Generated probabilities for jurir_no: {row['jurir_no']}")
        
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
        logger.info(f"Prediction generated for jurir_no: {row['jurir_no']}")

    logger.debug(f"Size of jurir_no_list: {len(jurir_no_list)}")
    logger.debug(f"Size of predictions: {len(predictions)}")
    logger.debug(f"Generated predictions: {predictions}")

    logger.info("Completed generate_predictions_dictionary function")
    return None, predictions

# view_DB_predict 에서 사용
def get_db_predictions(db: Session, model_info: str, page: int, page_size: int, search_query: str):
    if search_query == "None" or not search_query:
        search_query = ""  # 빈 문자열로 설정
    offset = (page - 1) * page_size

    search_filter = ""
    if search_query:
        search_filter = f"AND (pr.company_name LIKE :search_query OR pr.corporate_number LIKE :search_query)"

    query = text(f"""
    SELECT 
        pr.corporate_number AS "corporate_number", 
        pr.company_name AS "company_name", 
        pr.AAA AS "AAA", 
        pr.AA_plus AS "AA+", 
        pr.AA AS "AA", 
        pr.AA_minus AS "AA-", 
        pr.A_plus AS "A+", 
        pr.A AS "A", 
        pr.A_minus AS "A-", 
        pr.BBB_plus AS "BBB+", 
        pr.BBB AS "BBB", 
        pr.BBB_minus AS "BBB-", 
        pr.BB_plus AS "BB+", 
        pr.BB AS "BB", 
        pr.BB_minus AS "BB-", 
        pr.B_plus AS "B+", 
        pr.B AS "B", 
        pr.B_minus AS "B-" 
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
    AND pr.model_reference = :model_info
    {search_filter}
    ORDER BY pr.company_name
    LIMIT :limit OFFSET :offset
    """)

    total_query = text(f"""
    SELECT COUNT(*) FROM predict_ratings pr
    WHERE pr.corporate_number IN (
        SELECT jurir_no 
        FROM (
            SELECT DISTINCT a.jurir_no, b.totalAsset2023
            FROM companyInfo a
            LEFT JOIN FS2023 b ON a.jurir_no = b.jurir_no
            WHERE b.totalAsset2023 > 0
        ) AS subquery
    )
    AND pr.model_reference = :model_info
    {search_filter}
    """)

    result = db.execute(query, {"model_info": model_info, "limit": page_size, "offset": offset, "search_query": f"%{search_query}%"})
    predictions = result.fetchall()
    columns = result.keys()

    # Fetch the total number of items
    total_items_result = db.execute(total_query, {"model_info": model_info, "search_query": f"%{search_query}%"}).fetchone()
    total_items = total_items_result[0] if total_items_result else 0

    predictions_dict = [dict(zip(columns, row)) for row in predictions]
    return predictions_dict, total_items



def custmomized_train_model(file_path, selected_features, label_column, model_type, n_estimators=150, test_size=0.2, random_state=42, use_automl=False):
    # CSV 파일을 읽어오기
    df_origin = pd.read_csv(file_path)
    df = df_origin.dropna()

    # 입력 변수와 출력 변수 선택
    X = df[selected_features]
    y = df[label_column]

    # 데이터 스케일링 (선택사항)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    
        # 데이터 불균형 확인
    logger.debug(file_path)
    logger.debug("Original dataset shape %s" % Counter(y))
    logger.debug("X %s" % X)
    logger.debug("y %s" % y)

    
    # 데이터 불균형 처리 (SMOTE 사용)
    # if not use_automl:
    #     smote = SMOTE(random_state=random_state)
    #     X_res, y_res = smote.fit_resample(X_scaled, y)
    # else:
    #     X_res, y_res = X_scaled, y

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size, random_state=random_state)

    # AutoML 사용 여부에 따른 모델 학습
    # if use_automl:
    #     model = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=random_state, n_jobs=-1)
    # else:
        # 모델 선택 및 초기화
    if model_type == "randomforest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    elif model_type == "svm":
        model = SVC(random_state=random_state)
    elif model_type == "logistic":
        model = LogisticRegression(random_state=random_state)
    elif model_type == "knn":
        model = KNeighborsClassifier()
    else:
        raise ValueError("Unsupported model type.")

    # 모델 훈련
    logger.debug("Model training start")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"Model training completed in {elapsed_time:.2f} seconds")

    # 모델 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-'])

    # 모델 정보 저장
    model_info = {
        "model_name": model.__class__.__name__,
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_estimators": n_estimators if hasattr(model, 'n_estimators') else None,
        "max_features": getattr(model, 'max_features', None),
        "feature_importances": dict(zip(selected_features, model.feature_importances_)) if hasattr(model, 'feature_importances_') else None,
        "feature_names": selected_features,
        "n_samples": len(df)
    }

    return model, scaler, accuracy, class_report, conf_matrix, model_info, df_origin, selected_features

def set_default_model(db: Session, model_id: str):
    try:
        # 모든 모델의 is_default를 False로 초기화
        db.query(ModelStorage).update({ModelStorage.is_default: False})
        db.commit()

        # 선택한 model_id의 is_default를 True로 변경
        model = db.query(ModelStorage).filter(ModelStorage.model_id == model_id).first()
        if model:
            model.is_default = True
            db.commit()
            return True
        else:
            return False
    except SQLAlchemyError as e:
        db.rollback()
        raise e
    
    
    
def get_default_model(db: Session) -> Union[str, None]:
    try:
        # 기본값으로 설정된 모델을 조회
        model = db.query(ModelStorage).filter(ModelStorage.is_default == True).first()
        if model:
            return model.model_id
        else:
            return None
    except SQLAlchemyError as e:
        db.rollback()
        raise e
    
