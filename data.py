# ================================
# PHISHING URL DETECTION PROJECT
# Random Forest Regression
# ================================

print("=== PROGRAM STARTED ===")

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ================================
# STEP 1: Load Dataset
# ================================
data = pd.read_csv("malicious_phish.csv")

print("Dataset loaded successfully")
print(data.head())
print("Columns:", data.columns)

# ================================
# STEP 2: Convert Labels
# benign -> 0.0
# others -> 1.0
# ================================
data['label'] = data['type'].apply(
    lambda x: 0.0 if x == 'benign' else 1.0
)

print("Labels converted")
print(data[['type', 'label']].head())

# ================================
# STEP 3: Feature Extraction
# ================================
def extract_features(url):
    url_length = len(url)
    dot_count = url.count('.')
    has_ip = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    has_https = 1 if url.startswith("https") else 0

    return pd.Series([url_length, dot_count, has_ip, has_https])

# ================================
# STEP 4: Create Final Dataset
# (5 columns only)
# ================================
data[['url_length', 'dot_count', 'has_ip', 'has_https']] = data['url'].apply(extract_features)

final_data = data[['url_length', 'dot_count', 'has_ip', 'has_https', 'label']]

print("Final dataset created")
print(final_data.head())

# ================================
# STEP 5: Train-Test Split
# ================================
X = final_data.drop('label', axis=1)
y = final_data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split completed")

# ================================
# STEP 6: Train Random Forest Regressor
# ================================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Random Forest Regression model trained")

# ================================
# STEP 7: Calculate Approximate Accuracy
# ================================
y_pred_scores = model.predict(X_test)

# Convert regression output to class using threshold 0.7
y_pred_class = [1 if score >= 0.7 else 0 for score in y_pred_scores]

correct = 0
for pred, actual in zip(y_pred_class, y_test):
    if pred == actual:
        correct += 1

accuracy = correct / len(y_test)

print("Approximate Accuracy:", round(accuracy * 100, 2), "%")

# ================================
# STEP 8: User URL Prediction
# ================================
def predict_url(url):
    features = extract_features(url)

    feature_df = pd.DataFrame(
        [features],
        columns=['url_length', 'dot_count', 'has_ip', 'has_https']
    )

    risk_score = model.predict(feature_df)[0]

    print("\nURL:", url)
    print("Risk Score:", round(risk_score, 2))

    if risk_score >= 0.7:
        print("Fake / Phishing URL")
    else:
        print("Safe URL")

# ================================
# STEP 9: Take User Input
# ================================
user_url = input("\nEnter a URL to check: ")
predict_url(user_url)
