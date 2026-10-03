# 💻 Laptop Price Predictor

Predict a laptop's price (₹) from its specs, and group laptops into market segments using only their hardware.

**Results on 255 held-out laptops**

| Model | R² (log price) | MAE | MAPE |
|---|---|---|---|
| Linear Regression (baseline) | 0.815 | ₹12,749 | 21.5% |
| **XGBoost, tuned with RandomizedSearchCV** | **0.913** | **₹9,152** | **14.8%** |

## What's inside
| File | What it is |
|---|---|
| [`laptop_price_predictor.ipynb`](laptop_price_predictor.ipynb) | Full walkthrough with outputs: EDA → cleaning → feature engineering → models → tuning → clustering |
| [`features.py`](features.py) | Cleaning + feature engineering, shared by the notebook and the app |
| [`app.py`](app.py) | Streamlit app: choose specs → price, likely range and market segment |

## Approach
1. **EDA.** Price is right-skewed (skew 1.52), so the model predicts `log(Price)` (skew −0.17).
2. **Cleaning.** Dropped 29 duplicate rows. Converted `"8GB"` and `"1.37kg"` to numbers.
3. **Feature engineering.** Turned 12 messy columns into 13 clean features: **PPI** (from resolution + screen size), Touchscreen/IPS flags, CPU family + GHz, SSD vs HDD in GB, GPU brand, OS group.
4. **Linear Regression built from scratch with gradient descent.** It reaches the same R² (0.8369) as scikit-learn.
5. **Overfitting vs underfitting.** A decision tree depth sweep shows the bias–variance trade-off.
6. **Model race with 5-fold CV.** Linear, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost.
7. **Tuning.** GridSearchCV (Random Forest) and RandomizedSearchCV (XGBoost), plus a stacking ensemble.
8. **Unsupervised.** K-Means found 4 segments from specs alone (Budget ₹39k → Gaming ₹109k avg) without ever seeing the price. PCA for 2D plots. DBSCAN flagged 20 unusual configurations.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook laptop_price_predictor.ipynb   # run all cells -> creates model.joblib
streamlit run app.py
```

Data: [laptop_data.csv](https://github.com/campusx-official/laptop-price-predictor-regression-project), 1,303 laptops scraped from an Indian e-commerce site (~2018 prices).

## Resume bullet
> Built a laptop price predictor (1.3k listings) with engineered features (PPI, CPU tier, SSD/HDD split) and a tuned XGBoost model: **test R² 0.91, 28% lower error than a linear baseline** (MAE ₹9.2k vs ₹12.7k). Implemented gradient-descent linear regression from scratch, segmented the market with K-Means/PCA, and deployed a Streamlit app.
