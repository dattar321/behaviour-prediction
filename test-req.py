import requests

# Define the URL of the API
url = "http://127.0.0.1:8000/predict"

# Define the payload (example customer data)
payload = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 5.6,
    "TotalCharges": 74.4
}

# Make the POST request with the JSON payload
try:
    response = requests.post(url, json=payload)

    # Check the response status
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print("Error:", response.status_code, response.text)

except Exception as e:
    print("An error occurred:", e)
