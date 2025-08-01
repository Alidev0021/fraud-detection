# fraud-detection


🚀 Fraud Detection API with Tribal Name Validation

1. Install dependencies:
   pip install -r requirements.txt

2. Run the API:
   python app.py

3. Send POST request to:
   http://<your-ip>:5000/fraud-score

4. JSON Body format:
{
  "amount": 12000,
  "hour": 3,
  "device": "new",
  "last_name": "Xoni"
}

5. If last_name is not in tribal_names.csv, risk increases.
