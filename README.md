# 💰 Term Deposit Subscription Predictor

This project predicts whether a client will subscribe to a term deposit based on marketing data from a Portuguese bank. Built using Python and Jupyter Notebook, the project walks through the full data science pipeline from data wrangling to model evaluation.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository  
- **File Used:** `bank-additional.csv`  
- **Size:** 41,188 records × 21 features  
- **Target Variable:** `y` – whether the client subscribed to a term deposit (`yes`/`no`)

---

## 🧠 Features Used

Some examples of features in the dataset:

- Client Info: `age`, `job`, `marital`, `education`, `housing`, `loan`  
- Contact: `contact`, `month`, `day_of_week`  
- Campaign: `campaign`, `pdays`, `previous`, `poutcome`  
- Economic Context: `emp.var.rate`, `cons.price.idx`, `euribor3m`

---

## ⚙️ Project Workflow

1. **📥 Load & Explore Data**  
   - Used Pandas for EDA, summary stats, and data cleaning

2. **🧹 Preprocessing**  
   - One-hot encoding categorical variables  
   - Handled class imbalance (if applicable)

3. **📈 Modeling**  
   - Logistic Regression (baseline)  
   - Evaluated with accuracy, precision, recall, and F1-score

4. **📉 Evaluation**  
   - Confusion matrix  
   - Classification report  
   - ROC Curve

---

## 📁 Project Structure

term_deposit_sub_predictor/
├── bank-additional.csv
├── term_deposit_model.ipynb # Jupyter notebook with EDA and ML
├── README.md


---

## 🚀 How to Run

1. Clone the repo  
2. Open the `term_deposit_model.ipynb` in Jupyter  
3. Run all cells to reproduce results

---

## 📌 Key Takeaways

- Logistic Regression is a solid baseline model for binary classification tasks  
- Preprocessing and feature selection heavily impact performance  
- Bank marketing data provides strong business use cases for predictive modeling

---


