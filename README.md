# Phishing URL Detection System

A Machine Learning based web application that detects phishing (fake) URLs using a **Random Forest Classifier**.

This project analyzes structural URL features and classifies URLs as:

- ✅ Safe  
- ❌ Phishing  

Built using **Python, Scikit-learn, and Streamlit**.


## Project Description

Phishing attacks are one of the most common cyber security threats. Attackers create fake URLs that look like legitimate websites to steal user credentials and sensitive data.

This project uses a Machine Learning model to analyze URL structure and determine whether a URL is safe or malicious.

The application is built with Streamlit to provide an interactive web interface where users can:

- Enter a URL  
- Get instant prediction  
- View prediction accuracy  
- View URL check history  
- Remove URLs from history  

## How The System Works

1. The dataset (`malicious_phish.csv`) is loaded.
2. Labels are converted:
   - `benign` → 0 (Safe)
   - Others → 1 (Phishing)
3. The system extracts structural features from each URL:
   - URL Length
   - Dot Count
   - IP Address Presence
   - HTTPS Usage
4. Data is split into training and testing sets.
5. A **Random Forest Classifier** is trained.
6. The model predicts whether a user-entered URL is Safe or Phishing.
7. The result is displayed in the Streamlit web interface.

## Model Information

- Algorithm: Random Forest Classifier
- Train-Test Split: 80% Training / 20% Testing
- Accuracy: ~77% (may vary slightly due to random split)
- Features Used:
  - URL Length
  - Dot Count
  - IP Address Presence
  - HTTPS Usage

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Regular Expressions (re)

## Project Structure

detection_fake_url-s/
│
├── app.py
├── malicious_phish.csv
├── requirements.txt
└── README.md
