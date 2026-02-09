import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🔒",
    layout="wide"
)

st.title("🔒 Phishing URL Detection System")
st.write(
    "This system uses **Random Forest Classifier** to detect phishing URLs "
    "based on URL structural features."
)

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    data = pd.read_csv("malicious_phish.csv")
    data["label"] = data["type"].apply(lambda x: 0 if x == "benign" else 1)
    return data

data = load_data()

# ================================
# FEATURE EXTRACTION
# ================================
def extract_features(url):
    return [
        len(url),
        url.count("."),
        1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0,
        1 if url.startswith("https") else 0,
    ]

# ================================
# PREPARE DATA
# ================================
X = data["url"].apply(extract_features).tolist()
X = pd.DataFrame(X, columns=["url_length", "dot_count", "has_ip", "has_https"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# TRAIN MODEL
# ================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test) * 100

# ================================
# SIDEBAR
# ================================
st.sidebar.header("⚙️ Model Details")
st.sidebar.write("**Algorithm:** Random Forest Classifier")
st.sidebar.write("**Accuracy:**", f"{accuracy:.2f}%")

st.sidebar.write("### Features Used")
st.sidebar.write("- URL Length")
st.sidebar.write("- Dot Count")
st.sidebar.write("- IP Address Presence")
st.sidebar.write("- HTTPS Usage")

# ================================
# SESSION STATE (HISTORY)
# ================================
if "history" not in st.session_state:
    st.session_state.history = []

# ================================
# URL CHECK UI
# ================================
st.subheader("🔍 Check a URL")

url_input = st.text_input(
    "Enter a URL",
    placeholder="https://example.com"
)

if st.button("Check URL"):
    if url_input.strip() == "":
        st.warning("Please enter a URL")
    else:
        features = extract_features(url_input)
        prediction = model.predict([features])[0]

        result = "Safe" if prediction == 0 else "Phishing"

        if prediction == 0:
            st.success("✅ Safe URL")
        else:
            st.error("❌ Phishing URL")

        # Save history
        st.session_state.history.append({
            "url": url_input,
            "result": result
        })

# ================================
# HISTORY SECTION
# ================================
st.divider()

show_history = st.checkbox("📜 Show / Hide URL History")

if show_history and st.session_state.history:
    st.subheader("📄 URL Check History")

    for index, item in enumerate(st.session_state.history):
        col1, col2, col3 = st.columns([6, 2, 2])

        col1.write(item["url"])
        col2.write(item["result"])

        if col3.button("❌ Remove", key=index):
            st.session_state.history.pop(index)
            st.experimental_rerun()

elif show_history:
    st.info("No history available yet.")

# ================================
# FOOTER
# ================================
st.markdown(
    "<hr><center>Mini Project | Cyber Security | develop by purna</center>",
    unsafe_allow_html=True
)
