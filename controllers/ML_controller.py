from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from database import SessionLocal
from services_def.ML_service import preprocess_and_predict_proba, train_model, get_new_data_from_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
machineLearning = APIRouter()
templates = Jinja2Templates(directory="templates")

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@machineLearning.get("/predict/", response_class=HTMLResponse)
async def predict(request: Request, db: Session = Depends(get_db)):
    jurir_no_list = ['1101110000086', '1101110002694', '1101110002959']
    new_data = get_new_data_from_db(db, jurir_no_list)
    if new_data.empty:
        raise HTTPException(status_code=404, detail="Data not found for the given jurir_no list")
    
    model, scaler, accuracy, class_report, conf_matrix, model_info = train_model()
    
    results = []
    for _, row in new_data.iterrows():
        data = row.to_frame().T
        probabilities = preprocess_and_predict_proba(data, model, scaler)
        class_probabilities = list(zip(model.classes_, probabilities[0]))
        class_probabilities.sort(key=lambda x: x[1], reverse=True)
        
        cumulative_prob = 0.0
        sorted_probabilities = []
        for cls, prob in class_probabilities:
            cumulative_prob += prob
            sorted_probabilities.append({
                'class': cls, 
                'probability': round(prob, 2), 
                'cumulative': round(cumulative_prob, 2)
            })
        
        results.append({
            "jurir_no": row["jurir_no"],
            "sorted_probabilities": sorted_probabilities
        })
    
    # 피처 중요도 리스트를 feature name과 함께 zip하여 딕셔너리로 변환
    feature_importances = dict(zip(model.feature_names_in_, model_info['feature_importances']))

    return templates.TemplateResponse("ML_template/ML_view.html", {
        "request": request,
        "results": results,
        "accuracy": round(accuracy, 2),
        "class_report": class_report,
        "conf_matrix": conf_matrix,
        "model_info": {**model_info, "feature_importances": feature_importances}
    })
