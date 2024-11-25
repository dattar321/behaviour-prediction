import os, sys
from src.exception import CustomException
from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline

def predict_churn():
    try:
        # Sample data
        customer_data = CustomData(
            gender="Female",
            SeniorCitizen=0,
            Partner="Yes",
            Dependents="No",
            tenure=24,
            PhoneService="Yes",
            MultipleLines="No",
            InternetService="DSL",
            OnlineSecurity="Yes",
            DeviceProtection="No",
            TechSupport="Yes",
            StreamingTV="No",
            StreamingMovies="No",
            Contract="Month-to-month",
            PaperlessBilling="Yes",
            PaymentMethod="Electronic check",
            MonthlyCharges=65.6,
            TotalCharges=1574.4
        )
        
        # Convert to DataFrame
        df = customer_data.get_data_as_dataframe()
        print("DataFrame Created")
        
        # Initialize prediction pipeline
        predict_pipeline = PredictionPipeline()
        print("Prediction Pipeline Initialized")
        
        # Make prediction
        results = predict_pipeline.predict(df)
        
        return results
        
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        prediction = predict_churn()
        if prediction[0] == 1:
            print("Customer is likely to churn")
        else:
            print("Customer is likely to stay")
            
    except Exception as e:
        print(f"Error occurred: {e}")