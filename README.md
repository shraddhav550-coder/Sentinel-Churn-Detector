# Sentinel-Churn-Detector

Customer Churn Prediction & Intelligence Dashboard

Sentinel Churn Detector is a machine learning project that analyzes customer information to estimate the likelihood of customer churn. It combines a Scikit-learn classification pipeline with an interactive Streamlit dashboard to make churn analysis easier to explore and understand.

The project demonstrates how data preprocessing, exploratory data analysis, machine learning, and interactive visualization can work together to support customer-retention decisions.

 Project Overview

Customer churn occurs when customers stop using a company's products or services. Understanding which customers may be at risk can help businesses investigate customer behavior and plan retention strategies.

Sentinel uses customer demographics, service subscriptions, contract details, tenure, and billing information to predict churn probability.

Project goals:
- Build a machine learning model to predict customer churn.
- Explore historical customer data and churn patterns.
- Provide an interactive dashboard for customer-level predictions.
- Present insights that can help inform customer-retention strategies.

 Features

 Interactive Dashboard
- View key customer and churn metrics.
- Explore churn distribution and customer billing patterns.
- Compare churn across contract types.
- Filter customer records to investigate specific segments.

 Machine Learning Predictions
- Estimate churn probability for an individual customer.
- Display the prediction through visual indicators.
- Provide suggested retention actions based on configurable probability thresholds.
- Export prediction results as a CSV file.

 Customer Insights
- Explore customer tenure and monthly charges.
- Review customer records and contract summaries.
- Download filtered data for further analysis.

 Data Preparation
- Convert `TotalCharges` into a numeric format.
- Identify missing values and duplicate records.
- Prepare data for model training.

 Machine Learning Workflow

The project follows this workflow:

1. **Data collection:** Load the Telco Customer Churn dataset.
2. **Data cleaning:** Handle missing values and convert billing data to numeric format.
3. **Exploratory data analysis:** Examine churn patterns, customer tenure, contract types, and charges.
4. **Preprocessing:** Encode categorical variables and scale numerical features using a Scikit-learn pipeline.
5. **Model training:** Train a Random Forest classifier.
6. **Prediction:** Estimate the probability of churn for customer profiles.
7. **Visualization:** Present historical insights and predictions in Streamlit.

Project Structure

Sentinel-Churn-Detector/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.joblib
│
└── src/
    ├── eda.py
    └── train_model.py

**Note:** The dataset is excluded from this public repository through `.gitignore`. Download it from the source described below and place it in the `data/` folder before running the project. The trained model is included in the repository.

 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data cleaning and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing and machine learning |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| Plotly | Interactive charts |
| Streamlit | Interactive web dashboard |
| Joblib | Saving and loading the trained model |
| Git & GitHub | Version control and project hosting |

 Dataset

**Dataset:** IBM Telco Customer Churn

The dataset contains customer information such as:
- Customer demographics
- Services subscribed to
- Contract type and payment method
- Customer tenure
- Monthly and total charges
- Churn status

The original dataset contains 7,043 customer records and 21 columns. The preprocessing workflow converts `TotalCharges` to numeric format and removes records with missing essential values.

**Dataset source:** [IBM Sample Data — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Download the CSV and place it at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Check the dataset's license and terms before redistributing it.

 Installation & Setup

 1. Clone the repository

```bash
git clone https://github.com/shraddhav550-coder/Sentinel-Churn-Detector.git
cd Sentinel-Churn-Detector
```

2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

4. Add the dataset

Create a folder named `data` if it does not exist, then place the downloaded CSV inside it using the filename shown in the Dataset section.

5. Launch the dashboard

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually `http://localhost:8501`. Open that URL in your browser.

 Training the Model

The project includes scripts for data preparation, exploratory analysis, and model training.

Run the data-cleaning script:

```bash
python train_model.py
```

Run exploratory data analysis:

```bash
python src/eda.py
```

Train the model:

```bash
python src/train_model.py
```

The training script should generate or update the saved model at `models/churn_model.joblib`. Ensure the dataset is in place before running these commands.

 Model & Evaluation

The project uses a **Random Forest Classifier** within a Scikit-learn preprocessing pipeline. The configuration includes 200 trees, a fixed random seed, and balanced class weights.

The model produces a churn probability for a customer profile. This probability is an estimate, not a guarantee that the customer will leave.

Model performance should be evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

**Evaluation note:** Quantitative performance results should be added after running the evaluation on a held-out test set. No specific accuracy or performance score is claimed here.

 Business Applications

Potential uses of churn analysis include:
- Identifying customer groups that may need additional support.
- Reviewing churn patterns across contract types.
- Investigating relationships between tenure, charges, and churn.
- Helping customer-success teams prioritize retention outreach.
- Supporting data-driven discussions about customer experience.

These are potential applications, not proof that the model improves retention outcomes. Prediction quality and business impact should be validated before real-world deployment.

Current Scope

- The application is designed as a learning and portfolio project.
- Customer records and predictions are not automatically synchronized across devices.
- Retention suggestions are illustrative and should be validated before operational use.
- The model may be affected by class imbalance, data quality, and changes in customer behavior.

 Future Improvements

- Add a dedicated model evaluation page.
- Display feature importance and explain model predictions.
- Compare Random Forest with Logistic Regression and other classifiers.
- Add cross-validation and hyperparameter tuning.
- Improve testing, logging, and error handling.
- Deploy the dashboard online.
- Add monitoring to detect changes in model performance over time.

 Learning Outcomes

Through this project, I practiced:
- Cleaning and exploring a real-world-style dataset.
- Building a machine learning preprocessing pipeline.
- Training and saving a classification model.
- Integrating a trained model into a Streamlit application.
- Visualizing data with interactive charts.
- Organizing a Python project and managing it with Git and GitHub.


GitHub: [@shraddhav550-coder](https://github.com/shraddhav550-coder)

