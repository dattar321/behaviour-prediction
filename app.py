from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData

# Initialize the FastAPI app
app = FastAPI()

# Define the request schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
async def predict(data: CustomerData):
    try:
        # Convert request data into CustomData
        customer_data =   CustomData(
            gender=data.gender,
            SeniorCitizen=data.SeniorCitizen,
            Partner=data.Partner,
            Dependents=data.Dependents,
            tenure=data.tenure,
            PhoneService=data.PhoneService,
            MultipleLines=data.MultipleLines,
            InternetService=data.InternetService,
            OnlineSecurity=data.OnlineSecurity,
            DeviceProtection=data.DeviceProtection,
            TechSupport=data.TechSupport,
            StreamingTV=data.StreamingTV,
            StreamingMovies=data.StreamingMovies,
            Contract=data.Contract,
            PaperlessBilling=data.PaperlessBilling,
            PaymentMethod=data.PaymentMethod,
            MonthlyCharges=data.MonthlyCharges,
            TotalCharges=data.TotalCharges
        )

        # Convert to DataFrame
        customer_df = customer_data.get_data_as_dataframe()

        # Initialize prediction pipeline
        predict_pipeline = PredictionPipeline()

        # Make prediction
        prediction = predict_pipeline.predict(customer_df)

        # Return response
        return {
            "status": "success",
            "prediction": int(prediction[0]),
            "churn": "Yes" if prediction[0] == 1 else "No"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
