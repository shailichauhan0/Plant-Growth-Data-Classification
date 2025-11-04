🌱 Plant Growth Data Classification | Machine Learning Project

Predicting plant growth milestones using machine learning techniques to help optimize agricultural practices and greenhouse management.

Overview
This project focuses on predicting plant growth stages based on various environmental and management factors using machine learning.
It demonstrates the complete ML pipeline — from data preprocessing and feature engineering to model selection, hyperparameter tuning, and evaluation.

The dataset includes features such as:

Soil Type

Sunlight Hours

Water Frequency

Fertilizer Type

Temperature

Humidity

By analyzing these parameters, the model predicts the growth milestone (stage) of the plant.

Goal
To build a robust machine learning model capable of accurately classifying plant growth stages, enabling:

Smarter greenhouse management

Efficient irrigation and fertilizer usage

Data-driven decision-making for farmers

Dataset Details

Feature | Description | Type
Soil_Type | Type of soil (loam, sandy, clay, etc.) | Categorical
Sunlight_Hours | Average daily sunlight exposure | Numerical
Water_Frequency | Frequency of watering | Categorical
Fertilizer_Type | Fertilizer type (organic, chemical, none) | Categorical
Temperature | Average temperature (°C) | Numerical
Humidity | Relative humidity (%) | Numerical
Growth_Milestone | Target variable (0/1) | Integer

Samples: 193
Missing Values: None
Duplicates: None

Machine Learning Pipeline

Data Preprocessing

Checked for missing values and duplicates

Applied one-hot encoding to categorical features

Standardized numerical features using StandardScaler

Exploratory Data Analysis (EDA)

Visualized categorical and numerical feature distributions

Analyzed correlation between features and target variable

Identified trends using boxplots and countplots

Feature Engineering

Created 12+ derived and encoded features

Identified key predictors such as Temperature, Humidity, and Sunlight Hours

Model Training
Compared the performance of multiple machine learning algorithms:

Model | Accuracy
Random Forest | 62.07%
SVM | 60.34%
Decision Tree | 44.83%
KNN | 60.34%
Logistic Regression | 55.17%

Hyperparameter Tuning
Used GridSearchCV to optimize the Random Forest model parameters.

Best Parameters:
bootstrap: False
max_depth: 30
min_samples_leaf: 4
min_samples_split: 5
n_estimators: 200

✅ Final Accuracy: 94.83%
🔥 Improvement: from 62% → 94.8% after tuning

Visualizations

Count plots for categorical features

Histograms for numerical features

Boxplots showing relation between features and target variable

Feature importance visualization using Random Forest

Technologies Used
Language: Python
Data Manipulation: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-Learn
Environment: Google Colab

Project Structure

Plant-Growth-Data-Classification/
│
├── plant_growth_data.csv Dataset
├── data_preprocessing.py Data cleaning & preprocessing
├── eda_visualization.ipynb Exploratory Data Analysis
├── model_training.py Model training & evaluation
├── hyperparameter_tuning.py GridSearchCV tuning
├── README.md Project documentation
└── requirements.txt Dependencies (optional)

Results & Insights
Best Model: Random Forest Classifier
Final Accuracy: 94.8%
Key Predictors: Temperature, Humidity, Sunlight Hours
Use Case: Can be integrated into smart agriculture systems for plant monitoring and growth prediction.

Future Improvements

Apply Deep Learning (ANN/CNN) for larger datasets

Deploy as a Flask or Streamlit web app for live predictions

Collect larger and more diverse plant datasets for generalization

Integrate IoT sensors for real-time environmental data

Contact
👩‍💻 Shaili Chauhan
📧 shailichauhan06052004@gmail.com

Acknowledgement
This project was inspired by the concept of AI in Agriculture, aiming to enhance plant growth prediction and promote sustainable, data-driven farming practices. 🌿

If you found this project useful, please star this repository and share it with others!
