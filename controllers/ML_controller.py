from fastapi import APIRouter, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from database import SessionLocal
from services_def.ML_service import preprocess_and_predict_proba, train_model, get_new_data_from_db, insert_predictions_into_db, generate_predictions, generate_predictions_dictionary, get_db_predictions
import logging
import json
import asyncio
import time
from datetime import datetime

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

@machineLearning.get("/train/", response_class=HTMLResponse)
async def train(request: Request):
    username = request.session.get("username")
    model, scaler, accuracy, class_report, conf_matrix, model_info = train_model()
    
    # Define the credit ratings corresponding to each class
    credit_ratings = ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA', 'B', 'B+', 'B-', 'BB', 'BB+', 'BB-', 'BBB', 'BBB+', 'BBB-']

    
    # Store the model and related information
    model_store["model"] = model
    model_store["scaler"] = scaler
    model_store["accuracy"] = accuracy
    model_store["class_report"] = class_report
    model_store["conf_matrix"] = conf_matrix
    model_store["model_info"] = {**model_info, "feature_importances": dict(zip(model.feature_names_in_, model_info['feature_importances']))}

    return templates.TemplateResponse("ML_template/ML_view.html", {
        "request": request,
        "accuracy": round(accuracy, 2),
        "class_report": class_report,
        "conf_matrix": conf_matrix,
        "model_info": model_store["model_info"],
        "credit_ratings": credit_ratings,
        "show_predict_button": True,
        "username": username
    })


## 웹소켓 방식 시간 지연등 감안하여 사용안함.
# @machineLearning.get("/predict/", response_class=HTMLResponse)
# async def predict(request: Request):
#     return templates.TemplateResponse("ML_template/ML_creditViewWS.html", {"request": request})

# @machineLearning.websocket("/ws/predict/")
# async def websocket_predict(websocket: WebSocket, db: Session = Depends(get_db)):
#     await websocket.accept()
#     start_time = time.time()
#     try:
#         if model_store["model"] is None or model_store["scaler"] is None:
#             await websocket.send_text(json.dumps({"error": "Model not trained yet. Please train the model first."}))
#             await websocket.close()
#             return

#         error, predictions = generate_predictions(db, model_store["model"], model_store["scaler"])
        
#         if error:
#             await websocket.send_text(json.dumps(error))
#             await websocket.close()
#             return

#         for result in predictions:
#             await websocket.send_text(json.dumps(result))
#             await asyncio.sleep(0.1)

#         elapsed_time = time.time() - start_time
#         summary = {
#             "message": "completed",
#             "count": len(predictions),
#             "model_info": {
#                 "model_name": model_store["model_info"]["model_name"],
#                 "creation_date": model_store["model_info"]["creation_date"],
#                 "n_estimators": model_store["model_info"]["n_estimators"],
#                 "max_features": model_store["model_info"]["max_features"],
#                 "n_samples": model_store["model_info"]["n_samples"]
#             },
#             "elapsed_time": round(elapsed_time, 2)
#         }
#         await websocket.send_text(json.dumps(summary))

        
#         # Store the predictions in the global store
#         global predictions_store
#         predictions_store = predictions

#     except WebSocketDisconnect:
#         logger.info("WebSocket connection closed")
        

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

    return templates.TemplateResponse("ML_template/ML_creditViewHTML.html", {
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
    logger.info(predictions)
    
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




@machineLearning.get("/view_DB_predict/", response_class=HTMLResponse)
async def view_DB_predict(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    predictions_dict = get_db_predictions(db)
    # logger.info(predictions_dict)
    return templates.TemplateResponse("ML_template/ML_DBcreditView.html", {
        "request": request,
        "predictions": predictions_dict,
        "username": username
    })
    
    