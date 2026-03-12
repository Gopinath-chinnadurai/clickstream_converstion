# Customer Conversion Analysis Using Clickstream Data

### Project Overview

This project analyzes **customer browsing behavior (clickstream data)** to understand how users interact with an online shopping platform and predict whether a customer will complete a purchase.

The project applies **data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning models, and clustering techniques** to extract insights from user session data.

The entire workflow is implemented in a **Jupyter Notebook (.ipynb)**, which demonstrates the complete pipeline from **data loading and preprocessing to model training, evaluation, and customer segmentation**.

The dataset used in this project is the **UCI Online Shoppers Purchasing Intention (Clickstream) Dataset**.

---

## Objectives

The main objectives of this project are:

- Analyze customer browsing behavior
- Predict whether a customer will convert (make a purchase)
- Estimate potential revenue from user sessions
- Segment users based on browsing patterns
- Build a complete **machine learning pipeline using clickstream data**

---

## Dataset

The dataset used in this project is the **Online Shoppers Purchasing Intention Dataset** from the **UCI Machine Learning Repository**.

### Dataset Characteristics

- Session-based customer browsing data
- Multiple behavioral and interaction features
- Binary target variable indicating purchase conversion
- Suitable for classification and behavioral analysis

Example features include:

- Page views
- Time spent on pages
- Bounce rates
- Exit rates
- Product category browsing
- Traffic source
- Visitor type

---

## Project Workflow

The project follows a structured **data science workflow**.

```
Clickstream Dataset (UCI)
        ↓
Data Loading
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Engineering
        ↓
Data Transformation & Encoding
        ↓
Train-Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Customer Segmentation (Clustering)
        ↓
Model Saving
        ↓
Deployment using Streamlit
```

---

## Data Preprocessing

Before training the models, several preprocessing steps were applied:

- Handling missing values
- Removing duplicates
- Data type correction
- Feature selection
- Encoding categorical variables

These steps ensure the dataset is **clean and suitable for machine learning models**.

---

## Exploratory Data Analysis (EDA)

EDA was performed to understand customer browsing behavior and identify important patterns.

Key analyses include:

- Session behavior analysis
- Conversion rate patterns
- Feature distributions
- Correlation analysis
- User interaction insights

Visualization libraries used:

- Matplotlib
- Seaborn
- Plotly

---

## Feature Engineering

New features were created to improve model performance.

Examples include:

- Session engagement metrics
- Interaction-based features
- Behavioral indicators

These engineered features help models better capture **customer purchase intent**.

---

## Machine Learning Models

Multiple machine learning models were used to predict **customer conversion**.

### Classification Models

Used to predict whether a user will complete a purchase.

Algorithms explored:

- Logistic Regression
- Decision Tree
- Random Forest

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1 Score

---

## Customer Segmentation

Customer segmentation was performed using **K-Means Clustering**.

This helps group customers based on their browsing behavior.

Benefits:

- Identify high-value customers
- Understand browsing patterns
- Improve marketing strategies

---

## Implementation

The entire project is implemented in a **Jupyter Notebook (.ipynb)** which includes:

- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Machine learning model training
- Model evaluation
- Clustering analysis
- Data visualization

This notebook demonstrates a **complete end-to-end machine learning workflow**.

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit

---

## How to Run the Project

### 1 Clone the Repository

```
git clone https://github.com/Gopinath-chinnadurai/Clickstream-Conversion-Analysis.git
```

### 2 Navigate to the Project Folder

```
cd Clickstream-Conversion-Analysis
```

### 3 Install Dependencies

```
pip install -r requirements.txt
```

### 4 Run the Notebook

```
jupyter notebook Clickstream_Conversion_Analysis.ipynb
```

Run the notebook cells sequentially to reproduce the full analysis.

---

## Applications

This project can be useful for:

- E-commerce customer behavior analysis
- Conversion rate optimization
- Customer segmentation
- Marketing strategy improvement
- Personalized recommendation systems

---

## Conclusion

This project demonstrates a complete **data science and machine learning workflow** using clickstream data.

It showcases:

- Data preprocessing and EDA
- Feature engineering
- Machine learning modeling
- Customer segmentation
- Interactive data visualization

The project highlights how **customer browsing behavior can be leveraged to predict purchase intent and improve business decision-making**.

---

## Author

**Gopinath Chinnadurai**

GitHub:  
https://github.com/Gopinath-chinnadurai
