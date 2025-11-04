# 🌱 Plant Growth Classification using Machine Learning

This project focuses on classifying and predicting plant growth stages based on environmental and management factors using various machine learning models. The main objective is to analyze plant growth data, identify key influencing features, and create a robust prediction model that can help in better yield estimation and growth monitoring.

## 🚀 Overview
The project involves data preprocessing, feature selection, model training, and hyperparameter tuning to achieve the best prediction accuracy. Multiple machine learning models such as Random Forest, Decision Tree, SVM, Logistic Regression, and KNN were compared to determine the optimal one for classification.

## 🧠 Workflow
1. Data Collection and Cleaning  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering and Selection  
4. Model Building and Comparison  
5. Hyperparameter Tuning and Evaluation  
6. Visualization and Interpretation of Results  

## ⚙️ Machine Learning Models Used
- Random Forest Classifier  
- Decision Tree Classifier  
- Support Vector Machine (SVM)  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  

## 🧩 Hyperparameter Tuning
Performed GridSearchCV on Random Forest to achieve optimal results with parameters such as:  
`n_estimators = 200`, `max_depth = 12`, `min_samples_split = 5`, `min_samples_leaf = 4`

✅ **Final Accuracy:** 94.83%  
🔥 **Improvement:** from 62% → 94.8% after tuning

---

## 📊 Visualizations
- 📈 Count plots for categorical features  
- 📉 Histograms for numerical features  
- 📦 Boxplots showing relation between features and target variable  
- 📊 Feature importance visualization using Random Forest  
- 🔍 Correlation heatmap between numerical variables  

---

## 🧰 Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Tools:** Jupyter Notebook, Google Colab  
- **Version Control:** Git & GitHub  

---

## 🗂️ Project Structure
```
├── data/
│   ├── plant_growth_data.csv
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── visualize.py
├── results/
│   ├── accuracy_report.csv
│   ├── feature_importance.png
├── README.md
```

---

## 🚧 Future Improvements
- Integration with IoT sensors for real-time plant data  
- Deployment as a web application for live prediction  
- Use of Deep Learning models (CNN/LSTM) for better performance  
- Dataset expansion with more diverse plant species and environmental factors  

---

## 📬 Contact
👩‍💻 **Author:** Shaili Chauhan  
📧 **Email:** shailichauhan06052004@gmail.com  

