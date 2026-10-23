# Laptop Price Prediction

## Project Overview

Laptop prices vary depending on brand, hardware specifications, and other features. This project explores the factors associated with laptop prices and develops machine learning models to predict laptop prices based on their specifications.

## Business Question

**Can laptop prices be predicted using their features and hardware specifications?**

## Dataset

- **Source:** [Global Laptop Selling Dataset — Kaggle](https://www.kaggle.com/datasets/mdriazhossain/global-laptop-selling-dataset)
- **Original dataset size:** 1,303 rows × 12 columns
- **Target variable:** `Price_euros`

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Data Cleaning

The following cleaning tasks were performed to prepare the dataset for analysis and modelling.

1. **Screen resolution:** Extracted display features, resolution width, and resolution height from `ScreenResolution`.
2. **RAM:** Removed the `GB` unit and converted RAM capacity to an integer.
3. **Memory:** Separated storage capacity from storage type and converted storage capacities to numeric values.
4. **CPU:** Extracted CPU brand, model, and clock speed into separate columns, converting clock speed to a numeric data type.
5. **GPU:** Extracted GPU brand and model into separate columns.
6. **Weight:** Removed the `kg` unit and converted weight to a numeric data type.
7. **Product:** Removed redundant hardware specifications embedded in product names while preserving laptop model names.
8. **Operating system:** Grouped operating systems into broader categories using the `OsFamily` feature and dropped the original `OpSys` column.
9. **Redundant columns:** Dropped the original `ScreenResolution`, `Memory`, `Cpu`, and `Gpu` columns after extracting their relevant features.
10. **Outliers:** Removed outliers from `Price_euros` to reduce the influence of extreme values on model training.

## Exploratory Data Analysis (EDA)

EDA was performed to understand feature distributions, identify relationships with laptop prices, and investigate potential multicollinearity.

### Key observations

- **Price distribution:** Laptop prices were right-skewed, with the mean exceeding the median. Most laptops were priced at the lower end of the distribution, while some high-priced laptops reached approximately €6,000.
- **Weight distribution:** Laptop weight was bimodal, with one peak around 1.2–1.4 kg and another around 2.0–2.3 kg. The mean and median were both approximately 2.04 kg.
- **Company:** Dell, Lenovo, and HP had similar median prices of approximately €900–€1,000, although their price distributions were wide and contained high-price outliers.
- **Less common companies:** Fujitsu, LG, and Huawei had very small sample sizes. Their distributions were insufficient to support reliable conclusions about typical prices.
- **Laptop type:** Prices varied across laptop categories. Netbooks had a median price of approximately €340, while workstations had a median of approximately €2,064.
- **RAM:** Higher RAM capacities were associated with higher median laptop prices in the dataset.
- **CPU brand:** Intel laptops had a typical price of approximately €1,000, compared with approximately €500 for AMD laptops.

### Multicollinearity and feature relationships

- `ResolutionWidth` and `ResolutionHeight` were highly correlated with `Pixels`, which represents total pixel count.
- `Inches` and `Weight` showed a strong correlation. `Weight` was retained because it had a stronger observed correlation with `Price_euros`.
- `Flash` and `Hybrid` storage features were predominantly zero, suggesting that these storage types were uncommon in the dataset and might contribute limited predictive information.

These observations informed the feature selection and engineering process.

## Feature Engineering and Selection

The following transformations were applied to prepare features for modelling.

- **Pixels:** Calculated as `ResolutionWidth × ResolutionHeight` to represent total display resolution.
- **OsFamily:** Created broader operating system categories from `OpSys`.
- **Feature selection:** Dropped `Inches` while retaining `Weight`, and dropped `ResolutionWidth` and `ResolutionHeight` after creating `Pixels`.

The selection decisions were informed by exploratory analysis and correlation patterns.

## Model Development

Three regression approaches were evaluated:

- **Linear Regression:** Used as the initial predictive model.
- **Ridge Regression:** Applied L2 regularization, with the regularization parameter tuned using cross-validation.
- **Lasso Regression:** Applied L1 regularization, with the regularization parameter tuned using cross-validation.

A baseline model was also used as a reference for evaluating whether the trained models improved upon a simple prediction strategy.

## Model Evaluation

The models were evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

### RMSE

| Model             | Train RMSE (€) | Test RMSE (€) | Gap (€) |
| ----------------- | -------------: | ------------: | ------: |
| Baseline          |         623.98 |        623.88 |   -0.10 |
| Linear Regression |          87.69 |        317.31 |  229.61 |
| Ridge (CV tuned)  |         109.74 |        243.45 |  133.70 |
| Lasso (CV tuned)  |         132.64 |        219.23 |   86.59 |

### MAE

| Model             | Train MAE (€) | Test MAE (€) | Gap (€) |
| ----------------- | ------------: | -----------: | ------: |
| Baseline          |        498.90 |       508.71 |    9.81 |
| Linear Regression |         44.45 |       204.72 |  160.27 |
| Ridge (CV tuned)  |         70.22 |       170.45 |  100.23 |
| Lasso (CV tuned)  |        101.94 |       150.99 |   49.05 |

\*The gaps are calculated as test error minus training error.

### Residual Diagnostics

The residual plot showed that the spread of residuals increased as predicted prices increased.

- At predicted prices of approximately €150–€800, residuals were concentrated within a relatively narrow range.
- Beyond approximately €1,000, the residual spread widened considerably, suggesting heteroscedasticity.

This indicates that the constant-variance assumption of ordinary linear regression may not hold.

### Proposed improvements

- **Log transformation:** Evaluate a logarithmic transformation of `Price_euros` to reduce right skewness and potentially stabilize residual variance.
- **Non-linear models:** Evaluate models such as Random Forest Regressor and Gradient Boosting Regressor to capture non-linear relationships between laptop specifications and prices.

Any improvements will be assessed using the same train-test split and evaluation metrics.

## Key Findings

- Laptop prices varied across brands, laptop categories, RAM capacities, and CPU brands.
- Several hardware features exhibited strong correlations, motivating feature selection.
- Regularized regression models improved test-set performance compared with the baseline.
- Lasso Regression achieved the lowest reported test MAE and RMSE among the evaluated regression models.
- Residual diagnostics indicated that prediction errors became more variable at higher predicted prices, suggesting an opportunity to improve the model.

## Conclusion

Among the evaluated models, **Lasso Regression performed best on the test set**, achieving an MAE of approximately €150.99 and an RMSE of approximately €219.23.

Compared with ordinary Linear Regression and Ridge Regression, Lasso produced lower test errors and a smaller train-test error gap. These results suggest that regularization improved generalization on unseen data.

However, the residual analysis indicated that prediction errors varied with predicted price. Further experimentation with a log-transformed target and non-linear models could improve predictive performance.

## Project Structure

```text
laptop-price-prediction/
├── data/
│   ├── global_laptop_selling_data.csv
│   ├── clean_global_laptop_selling_data.csv
│   └── modelling_clean_global_laptop_selling_data.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_data_exploration.ipynb
│   └── 03_price_prediction.ipynb
├── environment.yml
├── .gitignore
└── README.md
```

## How to Run the Project

1. Clone the repository.
2. Create the Conda environment.
3. Activate the environment.
4. Open Jupyter Notebook and run the notebooks in sequence.

```bash
git clone <your-repository-url>
cd laptop-price-prediction
conda env create -f environment.yml
conda activate <your-environment-name>
jupyter notebook
```

Replace the repository URL and environment name with your actual values.
