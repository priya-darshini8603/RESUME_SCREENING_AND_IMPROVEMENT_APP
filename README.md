# 🤖 Resume Screening and Improvement App

This project implements a **Machine Learning pipeline** for **automated resume screening and improvement suggestions** using **Natural Language Processing (NLP)** and classification models like **Decision Trees** and **K-Nearest Neighbors (KNN)**.

---

## 📊 Features

- Resume data preprocessing using spaCy
- Text vectorization using TF-IDF
- Job category classification using ML models
- Hyperparameter tuning using GridSearchCV
- Model evaluation with accuracy, confusion matrix & classification report
- Resume improvement logic (extendable)
- Pickle-based model saving for deployment

---

## 🛠️ Tech Stack

- Python 3
- Libraries:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `spacy`
  - `matplotlib`, `seaborn`
- Models:
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier

---

## 📁 Dataset

- **UpdatedResumeDataSet.csv**  
  A labeled dataset mapping resumes to predefined job categories.

---

## 🧠 How it Works

1. **Load and clean** the resume text
2. **Tokenize & vectorize** using spaCy + TF-IDF
3. Train classification models:
   - KNN
   - DecisionTreeClassifier
4. Evaluate performance using:
   - Confusion Matrix
   - Classification Report
   - Accuracy Score
5. Save the best model using `pickle`


