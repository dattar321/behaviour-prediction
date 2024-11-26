from flask import Flask, request, jsonify
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        json_data = request.get_json()

        # Create an instance of CustomData using JSON data
        customer_data = CustomData(
            gender=json_data.get("gender"),
            SeniorCitizen=int(json_data.get("SeniorCitizen")),
            Partner=json_data.get("Partner"),
            Dependents=json_data.get("Dependents"),
            tenure=int(json_data.get("tenure")),
            PhoneService=json_data.get("PhoneService"),
            MultipleLines=json_data.get("MultipleLines"),
            InternetService=json_data.get("InternetService"),
            OnlineSecurity=json_data.get("OnlineSecurity"),
            DeviceProtection=json_data.get("DeviceProtection"),
            TechSupport=json_data.get("TechSupport"),
            StreamingTV=json_data.get("StreamingTV"),
            StreamingMovies=json_data.get("StreamingMovies"),
            Contract=json_data.get("Contract"),
            PaperlessBilling=json_data.get("PaperlessBilling"),
            PaymentMethod=json_data.get("PaymentMethod"),
            MonthlyCharges=float(json_data.get("MonthlyCharges")),
            TotalCharges=float(json_data.get("TotalCharges"))
        )

        # Convert to DataFrame
        customer_df = customer_data.get_data_as_dataframe()

        # Initialize the prediction pipeline
        predict_pipeline = PredictionPipeline()

        # Make the prediction
        prediction = predict_pipeline.predict(customer_df)

        # Return the prediction result
        return jsonify({
            "status": "success",
            "prediction": int(prediction[0]),
            "churn": "Yes" if prediction[0] == 1 else "No"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
