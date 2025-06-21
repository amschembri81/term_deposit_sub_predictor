import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Term Deposit Dashboard", layout="wide")
st.title("💰 Term Deposit Subscription Predictor")

# Load model and data
model = joblib.load("model_resampled.pkl")
feature_names = joblib.load("feature_names.pkl")
y_test = joblib.load("y_test.pkl")
y_pred = joblib.load("y_pred.pkl")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Model Overview", "Top Features", "Customer Explorer", "Batch Prediction", "Segment Analysis"])

if section == "Model Overview":
    st.header("📈 Model Evaluation Metrics")
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

elif section == "Top Features":
    st.header("🔍 Top 10 Important Features")
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    top_n = st.slider("Select number of top features to view:", 5, 20, 10)
    fig, ax = plt.subplots()
    feat_importances.nlargest(top_n).plot(kind='barh', ax=ax)
    plt.title("Top Important Features")
    st.pyplot(fig)

elif section == "Customer Explorer":
    st.header("🧑‍💼 Explore Individual Prediction")
    input_data = {}
    for feature in feature_names:
        if 'age' in feature:
            input_data[feature] = st.slider(f"{feature}", 18, 95, 35)
        elif 'duration' in feature:
            input_data[feature] = st.slider(f"{feature}", 0, 5000, 300)
        elif 'emp.var.rate' in feature:
            input_data[feature] = st.slider(f"{feature}", -3.0, 1.5, 0.0)
        else:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction Result")
    st.markdown(f"### ✅ Subscribed" if prediction else "### ❌ Not Subscribed")
    st.markdown(f"**Confidence:** {proba:.2%}")

elif section == "Batch Prediction":
    st.header("📂 Batch CSV Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with customer data")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        predictions = model.predict(batch_df)
        batch_df['Prediction'] = np.where(predictions == 1, 'Yes', 'No')
        st.dataframe(batch_df)
        st.download_button("Download Predictions", batch_df.to_csv(index=False), "predictions.csv")

elif section == "Segment Analysis":
    st.header("📊 Segment Analysis")
    filter_age = st.slider("Filter by Age:", 18, 95, (25, 60))
    filter_contact = st.selectbox("Select Contact Type:", ["All", "cellular", "telephone"])

    # Load raw data to filter
    raw_df = pd.read_csv("bank-additional-full.csv", sep=';')
    seg_df = raw_df.copy()
    seg_df = seg_df[(seg_df['age'] >= filter_age[0]) & (seg_df['age'] <= filter_age[1])]
    if filter_contact != "All":
        seg_df = seg_df[seg_df['contact'] == filter_contact]

    st.write(f"Segment Size: {len(seg_df)} records")
    chart_data = seg_df['y'].value_counts().rename(index={"yes": "Subscribed", "no": "Not Subscribed"})
    st.bar_chart(chart_data)

st.markdown("---")
st.markdown("Built by Amanda Morrison · [LinkedIn](https://www.linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourprofile)")
