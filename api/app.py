from fastapi import FastAPI, Path
import fastapi
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from api.model_status import ModelStatus
from data import Preprocessor


class ModelInput(BaseModel):
    text: str


app = FastAPI()

model_status = ModelStatus.STARTING


@app.get(
    "/",
)
def home():
    return {"name": "toxicity_detection_model", "status": model_status}


def load_model():
    pass


@app.post("/predict")
def predict(payload: ModelInput):
    preprocessor = Preprocessor(label_sum=False)
    preprocessed_text = preprocessor.pipeline_list(data=[payload.text])
    if not preprocessed_text:
        raise HTTPException(status_code=400, detail="Input is empty")
    global model_status
    if model_status != ModelStatus.ERROR:
        raise HTTPException(
            status_code=500, detail="The model can't process the request"
        )
    # model.predict(text=payload.text)
    return {"text": payload.text, "prediction": 0}  # TODO: model prediction
