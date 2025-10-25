# 🏥 Patient Sickness Prediction System  

A **Flask-based web application** that uses **Machine Learning models** to predict possible sickness of patients based on their health data.  
This project demonstrates end-to-end **data preprocessing, model training, and deployment** with an interactive web UI.  

---

## 📌 Features  
- Predicts patient sickness/health condition using trained ML models  
- Flask-based web interface for easy interaction  
- Supports uploading patient health data (CSV/Excel)  
- Pre-trained model (`.pkl` files) included  
- User-friendly results page  

---

## 🛠️ Tech Stack  
- **Python 3**  
- **Flask** (for web framework)  
- **Pandas, NumPy** (for data handling)  
- **Scikit-learn** (for ML model)  
- **HTML, CSS** (for front-end templates)  

---

## 📂 Project Structure  

patient-sickness-prediction-system/
│── app.py                     # Main Flask app
│── train_model.py             # Training script
│── requirements.txt           # Python dependencies
│── patient_data.csv           # Sample dataset
│── Training.xlsx / Testing.xlsx # Training & testing data
│── templates/
│   ├── index.html             # Homepage
│   ├── result.html            # Prediction result page
│   ├── homepage.html          # Optional additional pages
│   ├── symptom_checker.html
│   └── Patient Health Prediction.html
│── model.pkl / encoders.pkl   # Saved ML models
│── README.md                  # Project documentation

---

## ⚙️ Installation & Setup  

1. Clone the repository  
  git clone https://github.com/siddharthrajak49-commits/patient-sickness-prediction-system.git
cd patient-sickness-prediction-system
#	Create virtual environment 
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
#	Install dependencies
 pip install -r requirements.txt
# run the flask app
python app.py
📊 Dataset
The dataset includes patient health parameters such as:
	•	Age
	•	Test results
	•	Other clinical features

The model is trained using scikit-learn and serialized with pickle for deployment.
The dataset used in this project includes patient health parameters such as age, test results, and other clinical features.
Model is trained using scikit-learn, and serialized with pickle for deployment.
🧑 Author
<img width="1470" height="956" alt="Screenshot 2025-10-25 at 2 36 16 PM" src="https://github.com/user-attachments/assets/03ef2669-4717-4fda-b08b-c708bc0125fb" />

👤 Snigdh Kumar
	•	GitHub: @siddharthrajak49-commits

⸻
License

This project is open-source under the MIT License.
---

### ✅ Next Step for You  
1. Copy above content → create a file named **`README.md`** in your project root folder.  
2. Save it → commit & push to GitHub:  
   ```bash
   git add README.md
   git commit -m "Added README.md"
   git push<img width="1470" height="956" alt="Screenshot 2025-08-31 at 11 36 14 PM" src="https://github.com/user-attachments/assets/eff36c52-fad1-468a-b5ce-147904f81981" />
<img width="1470" height="956" alt="Screenshot 2025-08-31 at 11 36 14 PM" src="https://github.com/user-attachments/assets/36668a79-5664-4939-90c7-fad3d73450a0" />
