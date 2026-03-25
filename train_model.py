import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---------------- LOAD DATA ----------------
data = pd.read_csv("data/house_data.csv")

print("Dataset Columns:", data.columns)

# ---------------- FEATURES ----------------
X = data[['size','bedrooms','age']]
y = data['price']

# ---------------- SCALING ----------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------
models = {
    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),

    "DecisionTree": DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=10
    )
}

best_model = None
best_score = float('-inf')   # IMPORTANT FIX

print("\n📊 MODEL PERFORMANCE\n")

# ---------------- TRAIN LOOP ----------------
for name, model in models.items():

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)

    print(f"{name}")
    print(f"   R2 Score : {round(r2,4)}")
    print(f"   MSE      : {round(mse,2)}")
    print("-"*40)

    # SELECT BEST MODEL
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

# ---------------- SAVE MODEL ----------------
os.makedirs("model", exist_ok=True)

pickle.dump(best_model, open("model/house_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print(f"\n🏆 Best Model: {best_name}")
print("✅ Model & Scaler saved successfully")