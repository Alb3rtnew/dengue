# dengue
# Dengue Cases Prediction — Random Forest Model

A machine learning project that predicts dengue cases in the 
Philippines using a Random Forest Regression model. The model 
is trained on a cleaned dataset containing dengue case records 
by month, year, and region, and evaluated using RMSE, R², and 
MAE metrics to measure prediction accuracy.

## Technologies Used
- Python 3
- pandas & numpy
- scikit-learn (RandomForestRegressor)
- matplotlib & seaborn

## Dataset
- File: `dengue_cases_cleaned_datasets.xlsx`
- Features: Month, Year, Region
- Target: Dengue_Cases
- Train/Test Split: 70/30

## How to Run
1. Clone the repository
2. Install dependencies:
   pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
3. Place the dataset file in the same folder as the script
4. Run the script:
   python dengue.py

## Output
- Evaluation metrics printed in the console (RMSE, R², MAE)
- Feature importance rankings
- Saved visualization: `dengue_rf_results.png`
