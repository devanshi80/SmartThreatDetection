# SmartThreatDetection

SmartThreatDetection is a machine learning pipeline to detect botnet (malicious) traffic from normal (benign) traffic using the **CICIDS2017** dataset. It explores multiple classification algorithms, performs hyperparameter tuning, and evaluates model performance using both metrics and visualizations. It also includes an optional `app.py` file for deployment or extension purposes.

---

## 📁 Project Structure

📦SmartThreatDetection
├── app.py # Optional script for deployment
├── modelprediction.ipynb # Main notebook for training and evaluation
├── trained_model.pkl # Saved model for deployment/inference
├── README.md # Project overview and usage instructions
├── requirements.txt # Python dependencies

---

## 📊 Models Implemented

- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- **Voting Classifier** (Ensemble of above models)

Each model is:
- Tuned using hyperparameter search (e.g., GridSearchCV)  
- Evaluated using accuracy, confusion matrix, classification report  
- Compared using performance plots  

---

## ⚙️ Pipeline Highlights

- Preprocessing of CICIDS2017 dataset  
- Label encoding for `BENIGN` and `Bot` traffic  
- Train-test split with stratification  
- Feature normalization using `StandardScaler`  
- Hyperparameter tuning  
- Visualization of model performance  
- Saving best model (`trained_model.pkl`) for future use  

---

## 🧠 Dataset: CICIDS2017

This project uses the [**CICIDS2017** dataset](https://www.unb.ca/cic/datasets/ids-2017.html) developed by the Canadian Institute for Cybersecurity.

> ⚠️ **Dataset Not Included**  
Due to file size limitations, the dataset must be downloaded manually.

### 📌 Instructions:
1. Download the dataset from: [CICIDS2017 Official Page](https://www.unb.ca/cic/datasets/ids-2017.html)  
2. Extract and preprocess the data (combine files into a usable CSV if needed)  
3. Update this line in `modelprediction.ipynb`:
```python
df = pd.read_csv("YOUR_DATASET.csv")  # Replace with actual file path

🚀 Getting Started
1. Clone the repository

git clone https://github.com/your-username/SmartThreatDetection.git
cd SmartThreatDetection

2. Install dependencies
pip install -r requirements.txt

3. Run the notebook
jupyter notebook modelprediction.ipynb

Developed by Devanshi Jain for academic, cybersecurity, and machine learning practice.
