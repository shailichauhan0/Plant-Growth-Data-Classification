# 🌱 Plant Growth Data Classification | Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

> Predicting plant growth milestones using machine learning techniques to help optimize agricultural practices and greenhouse management.

---

## 📘 Overview

This project focuses on **predicting plant growth stages** based on various **environmental and management factors** using machine learning.  
It demonstrates the complete ML pipeline — from **data preprocessing** and **feature engineering** to **model selection**, **hyperparameter tuning**, and **evaluation**.

The dataset includes features such as:
- 🌾 Soil Type  
- 🌞 Sunlight Hours  
- 💧 Water Frequency  
- 🌿 Fertilizer Type  
- 🌡️ Temperature  
- 💨 Humidity  

By analyzing these parameters, the model predicts the **growth milestone (stage)** of the plant.

---

## 🎯 Goal

To build a robust machine learning model capable of accurately classifying plant growth stages, enabling:
- Smarter greenhouse management  
- Efficient irrigation and fertilizer usage  
- Data-driven decision-making for farmers  

---

## 🧩 Dataset Details

| Feature | Description | Type |
|----------|--------------|------|
| Soil_Type | Type of soil (loam, sandy, clay, etc.) | Categorical |
| Sunlight_Hours | Average daily sunlight exposure | Numerical |
| Water_Frequency | Frequency of watering | Categorical |
| Fertilizer_Type | Fertilizer type (organic, chemical, none) | Categorical |
| Temperature | Average temperature (°C) | Numerical |
| Humidity | Relative humidity (%) | Numerical |
| Growth_Milestone | Target variable (0/1) | Integer |

- **Samples:** 193  
- **Missing Values:** None  
- **Duplicates:** None  

---

## ⚙️ Machine Learning Pipeline

### 🔹 Data Preprocessing
- Checked for missing values and duplicates  
- Applied one-hot encoding to categorical features  
- Standardized numerical features using `StandardScaler`  

### 🔹 Exploratory Data Analysis (EDA)
- Visualized categorical and numerical feature distributions  
- Analyzed correlation between features and target variable  
- Identified trends using boxplots and countplots  

### 🔹 Feature Engineering
- Created 12+ derived and encoded features  
- Identified key predictors such as Temperature, Humidity, and Sunlight Hours  

### 🔹 Model Training
Compared the performance of multiple machine learning algorithms:

| Model | Accuracy |
|--------|-----------|
| Random Forest | 62.07% |
| Support Vector Machine (SVM) | 60.34% |
| Decision Tree | 44.83% |
| K-Nearest Neighbors (KNN) | 60.34% |
| Logistic Regression | 55.17% |

---

## 🧮 Hyperparameter Tuning

Used **GridSearchCV** to optimize the Random Forest model parameters.

**Best Parameters:**
```python
{
  'bootstrap': False,
  'max_depth': 30,
  'min_samples_leaf': 4,
  'min_samples_split': 5,
  'n_estimators': 200
}

If you found this project useful, please ⭐ star this repository and share it with others!
