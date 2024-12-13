from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the model
model = joblib.load("app/model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define the input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post("/predict/")
def predict(input_data: IrisInput):
    # Convert input to model format
    features = [[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
