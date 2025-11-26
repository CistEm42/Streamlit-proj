import streamlit as st
import pickle as pk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI
import json
from groq import Groq

# client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# def explain_prediction_llm(prediction: str, features: dict):
#     client = Groq(api_key=st.secrets["GROQ_API_KEY"])

#     prompt = f"""
# You are an AI assistant helping users understand a machine learning prediction.

# Prediction: {prediction}
# User Features: {features}

# Explain:
# 1. Why the model made this prediction
# 2. What the result means
# 3. What the user should do next
# 4. Use simple, friendly language
# 5. Keep it short and clear
# """

#     response = client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[
#             {"role": "system", "content": "You explain ML predictions clearly."},
#             {"role": "user", "content": prompt},
#         ]
#     )

#     return response.choices[0].message.content

def clean_data():
    # Fetching and cleaning the data
    data = pd.read_csv("data/data.csv")
    data.drop(["Unnamed: 32", 'id'], axis=1, inplace=True)
    data['diagnosis']=data['diagnosis'].replace({'M':1,'B':0})
    data['diagnosis'] = data['diagnosis'].astype("category", copy=False)
    print(data.head())
    return data


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurement")
    data = clean_data()
    
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

def get_scaled_values(input_dict):
    data = clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)
    categories = ['Radius','Texture','Perimeter',
              'Area', 'Smoothness', 'Compactness', 'Concavity', 
              'Concave Points','Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],input_data['texture_mean'], input_data['perimeter_mean'], input_data['area_mean'],
            input_data['smoothness_mean'], input_data['compactness_mean'], input_data['concavity_mean'], input_data['concave points_mean'], 
            input_data['symmetry_mean'], input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'], input_data['concave points_se'], 
            input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],input_data['texture_worst'], input_data['perimeter_worst'], input_data['area_worst'],
            input_data['smoothness_worst'], input_data['compactness_worst'], input_data['concavity_worst'], input_data['concave points_worst'], 
            input_data['symmetry_worst'], input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )
    
    return fig


def add_predictions(input_data):
    model = pk.load(open("model/model.pkl", "rb"))
    scaler = pk.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)



    prediction = model.predict(input_array_scaled)[0]
    proba = model.predict_proba(input_array_scaled)[0]

    # Convert prediction to label
    prediction_label = "Benign" if prediction == 0 else "Malignant"

    st.subheader("Cell Cluster Prediction")
    st.write(f"The model predicts: **{prediction_label}**")

    probabilities = {
        "benign": float(proba[0]),
        "malignant": float(proba[1])
    }

    st.write("Benign probability:", probabilities["benign"])
    st.write("Malignant probability:", probabilities["malignant"])

    st.info("This tool assists doctors and should not replace professional diagnosis.")

    return prediction_label, probabilities
  

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor",
        layout="wide",
        initial_sidebar_state='expanded'
    )

    input_data = add_sidebar()
    #st.write(input_data)


    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer" \
        " from your tissue samples. This app predicts using a machine learning model whether a breast mass is" \
        "benign or malignant based on the measurements it receives from your cytosis lab. You can also update" \
        " the measurements by hand using the sliders in the sidebar"
                 )
    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        prediction_label, probabilities = add_predictions(input_data)

# Chatbot Section
    if st.button("Explain Prediction"):
        explanation = explain_prediction_llm(prediction_label, input_data)
        st.write(explanation)



if __name__ == '__main__':
    main()