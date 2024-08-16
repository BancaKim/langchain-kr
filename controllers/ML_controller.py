import os
import shutil
from fastapi import APIRouter, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import UploadFile, File
from sqlalchemy.orm import Session
from database import SessionLocal
from services_def.ML_service import train_model, insert_predictions_into_db, generate_predictions, generate_predictions_dictionary, get_db_predictions
from services_def.ML_service import save_model_info_to_db, get_all_model_info, get_model_info_by_id, custmomized_train_model, set_default_model
import logging

import time
from datetime import datetime
import os
import joblib
import pickle
import json

from typing import Dict, List
from io import BytesIO
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
machineLearning = APIRouter()
templates = Jinja2Templates(directory="templates")

model_store = {
    "model": None,
    "scaler": None,
    "accuracy": None,
    "class_report": None,
    "conf_matrix": None,
    "model_info": None,
}

# Global variable to store predictions
predictions_store = []

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
        


# @machineLearning.get("/train/", response_class=HTMLResponse)
# async def train(request: Request, db: Session = Depends(get_db)):
#     username = request.session.get("username")
#     model, scaler, accuracy, class_report, conf_matrix, model_info, df_origin, feature_columns = train_model()


#     # Define the credit ratings corresponding to each class
#     credit_ratings = ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA', 'B', 'B+', 'B-', 'BB', 'BB+', 'BB-', 'BBB', 'BBB+', 'BBB-']

#     # Store the model and related information
#     model_store = {
#         "model": model,
#         "scaler": scaler,
#         "accuracy": accuracy,
#         "class_report": class_report,
#         "conf_matrix": conf_matrix,
#         "model_info": {**model_info, "feature_importances": dict(zip(model.feature_names_in_, model_info['feature_importances']))}
#     }

#    # Save model info to DB
#     logger.info('Save model info to DB start')
#     save_model_info_to_db(
#         model_info=model_info,
#         accuracy=accuracy,
#         model=model,
#         scaler=scaler,
#         class_report=class_report,
#         conf_matrix=conf_matrix,
#         feature_columns=feature_columns,
#         db=db
#     )

#     return templates.TemplateResponse("ML_template/ML_view.html", {
#         "request": request,
#         "accuracy": round(accuracy, 2),
#         "class_report": class_report,
#         "conf_matrix": conf_matrix,
#         "model_info": model_info,
#         "credit_ratings": credit_ratings,
#         "show_predict_button": True,
#         "username": username
#     })
    
    
@machineLearning.get("/model_detail/{model_id}", response_class=HTMLResponse)
async def view_model_detail(request: Request, model_id: str, db: Session = Depends(get_db)):
    username = request.session.get("username")
    model_info = get_model_info_by_id(db, model_id)

    if not model_info:
        return HTMLResponse(content="Model not found", status_code=404)

    # Parse JSON fields from the database
    accuracy = model_info['accuracy']
    class_report = json.loads(model_info['class_report'])
    conf_matrix = json.loads(model_info['conf_matrix'])
    credit_ratings = ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA', 'B', 'B+', 'B-', 'BB', 'BB+', 'BB-', 'BBB', 'BBB+', 'BBB-']

    # Load the model and scaler if necessary
    model_filepath = model_info['model_filepath']
    model_store["model"] = joblib.load(model_filepath)
    model_store["scaler"] = pickle.loads(model_info['scaler'])
    model_store["accuracy"] = accuracy
    model_store["class_report"] = class_report
    model_store["conf_matrix"] = conf_matrix
    model_store["model_info"] = {**model_info, "feature_importances": model_info['feature_importances']}

    return templates.TemplateResponse("ML_template/ML_view.html", {
        "request": request,
        "accuracy": round(accuracy, 2),
        "class_report": class_report,
        "conf_matrix": conf_matrix,
        "model_info": model_store["model_info"],
        "credit_ratings": credit_ratings,
        "show_predict_button": True,
        "username": username,
        "model_id": model_id
    })
    
    




@machineLearning.get("/predict_all/", response_class=HTMLResponse)
async def predict_all(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    start_time = time.time()
    if model_store["model"] is None or model_store["scaler"] is None:
        raise HTTPException(status_code=400, detail="Model not trained yet. Please train the model first.")
    
    error, predictions = generate_predictions_dictionary(db, model_store["model"], model_store["scaler"])
    
    if error:
        raise HTTPException(status_code=400, detail=error)

    elapsed_time = round(time.time() - start_time, 2)
    model_info = {
        "model_name": model_store["model_info"]["model_name"],
        "creation_date": model_store["model_info"]["creation_date"],
        "n_estimators": model_store["model_info"]["n_estimators"],
        "max_features": model_store["model_info"]["max_features"],
        "n_samples": model_store["model_info"]["n_samples"]
    }
    # logger.info(predictions)
    return templates.TemplateResponse("/ML_template/ML_creditViewHTML.html", {
        "request": request,
        "predictions": predictions,
        "model_info": model_info,
        "elapsed_time": elapsed_time,
        "count": len(predictions),
        "show_db_button": True,
        "username": username 
    })


@machineLearning.post("/insert_predictions/", response_class=JSONResponse)
async def insert_predictions(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")    
    if model_store["model"] is None or model_store["scaler"] is None:
        return JSONResponse(content={"error": "Model not trained yet. Please train the model first."}, status_code=400)
    
    error, predictions = generate_predictions(db, model_store["model"], model_store["scaler"])
    
    if error:
        return JSONResponse(content=error, status_code=400)

    model_info = model_store["model_info"]
    creation_date = datetime.strptime(model_info['creation_date'], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
    model_reference = f"{round(model_store['accuracy'], 2)}_{model_info['model_name']}_{creation_date}"
    # logger.info(predictions)
    
    for prediction in predictions:
        # sorted_probabilities를 class 기준으로 정렬
        sorted_probabilities = sorted(prediction['sorted_probabilities'], key=lambda x: x['class'])
        probabilities_dict = {prob['class']: prob['probability'] for prob in sorted_probabilities}
        
        result = {
            "jurir_no": prediction["jurir_no"],
            "corp_name": prediction["corp_name"],
            "base_year": 2023,
            "AAA_plus": probabilities_dict.get('AAA+', 0.0),
            "AAA": probabilities_dict.get('AAA', 0.0),
            "AAA_minus": probabilities_dict.get('AAA-', 0.0),
            "AA_plus": probabilities_dict.get('AA+', 0.0),
            "AA": probabilities_dict.get('AA', 0.0),
            "AA_minus": probabilities_dict.get('AA-', 0.0),
            "A_plus": probabilities_dict.get('A+', 0.0),
            "A": probabilities_dict.get('A', 0.0),
            "A_minus": probabilities_dict.get('A-', 0.0),
            "BBB_plus": probabilities_dict.get('BBB+', 0.0),
            "BBB": probabilities_dict.get('BBB', 0.0),
            "BBB_minus": probabilities_dict.get('BBB-', 0.0),
            "BB_plus": probabilities_dict.get('BB+', 0.0),
            "BB": probabilities_dict.get('BB', 0.0),
            "BB_minus": probabilities_dict.get('BB-', 0.0),
            "B_plus": probabilities_dict.get('B+', 0.0),
            "B": probabilities_dict.get('B', 0.0),
            "B_minus": probabilities_dict.get('B-', 0.0),
            "CCC_plus": probabilities_dict.get('CCC+', 0.0),
            "CCC": probabilities_dict.get('CCC', 0.0),
            "CCC_minus": probabilities_dict.get('CCC-', 0.0),
            "C": probabilities_dict.get('C', 0.0),
            "D": probabilities_dict.get('D', 0.0),
            "model_reference": model_reference,
            "username": username
        }
        insert_predictions_into_db(db, result, model_reference)

    return JSONResponse(content={"message": "ML로 생성한 신용등급 추정치가 DB에 입력되었습니다."}, status_code=200)




@machineLearning.get("/modelControl/", response_class=HTMLResponse)
async def view_DB_predict(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    all_model_info = get_all_model_info(db)
    # logger.info(predictions_dict)
    return templates.TemplateResponse("ML_template/ML_ModelControl.html", {
        "request": request,
        "all_model_info" : all_model_info,
        "username": username
    })
    

@machineLearning.get("/ML_pretrain/", response_class=HTMLResponse)
async def ML_pretrain(request: Request, db: Session = Depends(get_db)):
    # username = request.session.get("username")
    # all_model_info = get_all_model_info(db)
    # # logger.info(predictions_dict)
    return templates.TemplateResponse("ML_template/ML_pretrain.html", {
        "request": request,
        # "all_model_info" : all_model_info,
        # "username": username
    })
    


@machineLearning.post("/customized_train/", response_class=HTMLResponse)
async def train(request: Request, file_upload: UploadFile = File(...), db: Session = Depends(get_db)):
    # 폴더 경로 설정
    save_directory = "saved_dataset"
    
    # 디렉토리가 없으면 생성
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 파일 저장 경로 설정
    file_path = os.path.join(save_directory, file_upload.filename)

    # 파일 저장
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file_upload.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail="File could not be saved.")

    # 비동기 방식으로 폼 데이터 가져오기
    form = await request.form()
    selected_features = form.getlist("selectedFeatures")  # 여러 선택 항목을 리스트로 가져오기
    label_column = form.get("selectedLabel")
    model_type = form.get("selectedModel")
    n_estimators = int(form.get("numTrees", 150))
    test_size = float(form.get("testSize", 0.2))
    random_state = int(form.get("randomState", 42))
    use_automl = form.get("useAutoML") == "True"

    # 신용 등급 정의
    credit_ratings = ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA', 'B', 'B+', 'B-', 'BB', 'BB+', 'BB-', 'BBB', 'BBB+', 'BBB-']

    # 커스텀 학습 모델 호출
    model, scaler, accuracy, class_report, conf_matrix, model_info, df_origin, selected_features = custmomized_train_model(
        file_path=file_path,
        selected_features=selected_features,
        label_column=label_column,
        model_type=model_type,
        n_estimators=n_estimators,
        test_size=test_size,
        random_state=random_state,
        use_automl=use_automl
    )

    # feature_importances를 지원하는 모델인지 확인
    if hasattr(model, "feature_importances_"):
        model_info["feature_importances"] = dict(zip(model.feature_names_in_, model.feature_importances_))
    else:
        model_info["feature_importances"] = None
        

    # Store the model and related information in the global model_store
    model_store["model"] = model
    model_store["scaler"] = scaler
    model_store["accuracy"] = accuracy
    model_store["class_report"] = class_report
    model_store["conf_matrix"] = conf_matrix
    model_store["model_info"] = {**model_info, "feature_importances": dict(zip(model.feature_names_in_, model_info['feature_importances']))}

        
    # Save model info to DB
    logger.info('Save model info to DB start')
    save_model_info_to_db(
        model_info=model_info,
        accuracy=accuracy,
        model=model,
        scaler=scaler,
        class_report=class_report,
        conf_matrix=conf_matrix,
        feature_columns=selected_features,
        db=db
    )

    # 결과 반환
    return templates.TemplateResponse("ML_template/ML_view.html", {
        "request": request,
        "accuracy": round(accuracy, 2),
        "class_report": class_report,
        "conf_matrix": conf_matrix,
        "model_info": model_info,
        "credit_ratings": credit_ratings,
        "show_predict_button": True,
        "username": request.session.get("username")
    })
    
    
@machineLearning.post("/default_model_pick/{model_id}")
async def default_model_pick(model_id: str, db: Session = Depends(get_db)):
    logger.info(model_id)
    success = set_default_model(db, model_id)
    
    
    model_info = get_model_info_by_id(db, model_id)
    
    # Parse JSON fields from the database
    accuracy = model_info['accuracy']
    class_report = json.loads(model_info['class_report'])
    conf_matrix = json.loads(model_info['conf_matrix'])

    # Load the model and scaler if necessary
    model_filepath = model_info['model_filepath']
    model_store["model"] = joblib.load(model_filepath)
    model_store["scaler"] = pickle.loads(model_info['scaler'])
    model_store["accuracy"] = accuracy
    model_store["class_report"] = class_report
    model_store["conf_matrix"] = conf_matrix
    model_store["model_info"] = {**model_info, "feature_importances": model_info['feature_importances']}
    
    logger.info(model_filepath)
        
    if not success:
        raise HTTPException(status_code=404, detail="Model not found or could not be set as default.")
    
    return {"success": True, "message": f"Model {model_id}이 디폴트 모델로 설정되었습니다."}