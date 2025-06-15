import streamlit as st
import joblib
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

# Load the trained model
model = joblib.load("best_knn_model.pkl")

# Simulated user database (email/password)
import json
import os

USERS_FILE = "users.json"

# Load users from file or initialize
if os.path.exists(USERS_FILE):
    with open(USERS_FILE, "r") as f:
        st.session_state.users = json.load(f)
else:
    st.session_state.users = {}

# Session state setup
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

HISTORY_FILE = "history.json"

# Load history from file or initialize
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        st.session_state.history = json.load(f)
else:
    st.session_state.history = []

# Registration page
if not st.session_state.authenticated:
    st.title("Register and Login to Wind Power AI Forecasting App")
    auth_tab = st.tabs(["Login", "Register"])

    with auth_tab[0]:
        with st.form("Login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

        if submit:
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.authenticated = True
                st.session_state.user = username
                st.rerun()
            else:
                st.error("Invalid email or password")

    with auth_tab[1]:
        with st.form("Register"):
            new_email = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            register = st.form_submit_button("Register")

        if register:
            import re
            password_valid = (
                len(new_password) >= 6 and
                re.search(r"[A-Z]", new_password) and
                re.search(r"[!@#$%^&*(),.?\":{}|<>]", new_password)
            )

            if not password_valid:
                st.error("Password must be at least 6 characters long, contain one uppercase letter, and one special character.")
            elif new_email in st.session_state.users:
                st.warning("User already exists.")
            else:
                st.session_state.users[new_email] = new_password
                with open(USERS_FILE, "w") as f:
                    json.dump(st.session_state.users, f)
                st.success("Registration successful! Please login above.")

# ---- Authenticated App ----
if st.session_state.authenticated:
    st.set_page_config(page_title=" Wind Power Forecasting AI", layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
        <style>
body {
    background-color: #f4f6f8;
    font-family: 'Segoe UI', sans-serif;
}
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 5px;
    color: #1e88e5;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    animation: fadeIn 1.5s ease-in-out;
}
.subtitle {
    text-align: center;
    font-size: 22px;
    font-weight: 400;
    color: #37474f;
    animation: fadeIn 2s ease-in-out;
}
.stButton>button {
    background-color: #1976d2;
    color: white;
    font-weight: 600;
    border-radius: 6px;
    padding: 0.5em 1.5em;
}
@keyframes fadeIn {
    0% {opacity: 0;}
    100% {opacity: 1;}
}
</style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">⚡ Wind Power Forecasting AI </div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Smart predictions. Real-time analytics. Professional insights.</div>', unsafe_allow_html=True)
    st.markdown(f"""---
Welcome, **{st.session_state.user}**!
""")

    selected = option_menu(None, ["Home", "Manual Input", "Upload CSV", "History", "About"],
                           icons=['house', 'sliders', 'upload', 'clock-history', 'info-circle'],
                           menu_icon="cast", default_index=0, orientation="horizontal")

    def predict_and_suggest(features):
        prediction = model.predict([features])[0]
        wind_speed, reactive_power, gearbox_temp, gen_wind2, gen_wind1 = features
        suggestions = []

        if wind_speed < 3:
            suggestions.append(" Wind speed is too low. Turbine may be in cut-in phase and not generating power.")
        elif wind_speed > 25:
            suggestions.append(" Wind speed exceeds safety limit. Turbine might auto-shutdown to prevent damage.")
        if reactive_power < 100:
            suggestions.append(" Reactive power too low. May affect voltage stability and grid compliance.")
        elif reactive_power > 900:
            suggestions.append(" Reactive power is high. Check capacitor banks or load imbalance.")
        if gearbox_temp > 85:
            suggestions.append(" Gearbox oil temperature is high. Risk of oil degradation or gearbox wear.")
        elif gearbox_temp < 30:
            suggestions.append(" Oil too cold. May increase mechanical resistance.")
        if gen_wind2 > 95:
            suggestions.append(" Generator winding 2 temperature too high. Risk of insulation failure.")
        if gen_wind1 > 95:
            suggestions.append(" Generator winding 1 temperature too high. Risk of insulation failure.")
        if not suggestions:
            suggestions.append("✅ Turbine conditions are optimal. Efficient power generation expected.")

        return prediction, suggestions

    if selected == "Manual Input":
        st.markdown(" Manual Input for Prediction")
        with st.form(key="input_form"):
            col1, col2 = st.columns(2)
            with col1:
                wind_speed = st.slider(" Wind Speed (m/s)", 0.0, 25.0, 10.0)
                reactive_power = st.slider(" Reactive Power (kVAR)", 0, 1000, 400)
                gearbox_temp = st.slider(" Gearbox Oil Temperature (°C)", 0, 120, 60)
            with col2:
                gen_wind2 = st.slider(" Generator Winding 2 Temp (°C)", 0, 120, 60)
                gen_wind1 = st.slider(" Generator Winding 1 Temp (°C)", 0, 120, 60)

            input_features = [wind_speed, reactive_power, gearbox_temp, gen_wind2, gen_wind1]
            submit = st.form_submit_button(" Predict Power Output")

        if submit:
            prediction, suggestions = predict_and_suggest(input_features)
            record = {"features": input_features, "prediction": prediction}
            st.session_state.history.append(record)
            with open("history.json", "w") as f:
                json.dump(st.session_state.history, f, indent=2)
            st.success(f"✅ Predicted Active Power: {prediction:.2f} kW")
            st.subheader("💡 Suggestions")
            for s in suggestions:
                st.info(s)

    elif selected == "History":
        st.markdown("📊 Prediction History Log")
        if st.session_state.history:
            for i, record in enumerate(reversed(st.session_state.history), 1):
                st.markdown(f"**#{i}** ➡️ Input: {record['features']} | 🔋 Prediction: {record['prediction']:.2f} kW")
        else:
            st.info("No prediction history yet.")

    elif selected == "Upload CSV":
        st.markdown("### 📁 Upload Turbine Dataset")
        file = st.file_uploader("Upload a CSV with columns: WindSpeed, ReactivePower, GearboxOilTemperature, GeneratorWinding2Temperature, GeneratorWinding1Temperature", type=["csv"])
        if file:
            df = pd.read_csv(file)
            required = ['WindSpeed', 'ReactivePower', 'GearboxOilTemperature', 'GeneratorWinding2Temperature', 'GeneratorWinding1Temperature']
            if all(col in df.columns for col in required):
                df[required] = df[required].interpolate(method='linear', limit_direction='both')
                df["Predicted Active Power"] = model.predict(df[required].values)
                for _, row in df.iterrows():
                    record = {
                        "features": [row[col] for col in required],
                        "prediction": row["Predicted Active Power"]
                    }
                    st.session_state.history.append(record)
                with open("history.json", "w") as f:
                    json.dump(st.session_state.history, f, indent=2)
                st.success("✅ Predictions generated successfully")
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
            else:
                st.error(f"❌ CSV must include the following columns: {required}")

    elif selected == "Home":
        st.image("https://static5.depositphotos.com/1015682/422/i/450/depositphotos_4224453-stock-photo-wind-turbine-farm-at-sunset.jpg", width=600)
        st.markdown("""
        Welcome to the Wind Power Forecasting AI App
        
        This application leverages artificial intelligence to predict wind turbine active power output in real-time.
        This application aims to assist engineers, analysts, and operators in predicting wind turbine power output using intelligent ML models.
        By leveraging historical and real-time data, users can forecast turbine performance, prevent overheating, and optimize energy output.

        
        How Do These Features Impact Power Generation?

        The following features are used to predict Active Power Output from wind turbines:

        - Wind Speed (m/s): Directly influences the kinetic energy available to the turbine. Low wind speed = low power; very high = safety shutdown.
        
        - Reactive Power (kVAR): Indicates how efficiently energy is transferred. Imbalances can affect voltage stability and turbine performance.
        
        - Gearbox Oil Temperature (°C): Too high indicates stress or friction; too low increases mechanical resistance. Optimal range means smooth power transfer.
        
        - Generator Winding 1 & 2 Temp (°C): Overheating reduces generator efficiency and risks damage. Proper cooling ensures maximum output.

        These five parameters together reflect mechanical, electrical, and environmental factors essential for safe and efficient wind power generation.

        
        Welcome to the Wind Power Forecasting AI App
        This application leverages artificial intelligence to predict wind turbine active power output in real-time.

        💡 Why this matters:
        - Optimizes wind turbine performance
        - Minimizes downtime and overheating risks
        - Helps with grid energy planning and sustainability

        ➤ You can manually input turbine parameters or upload a CSV file for batch predictions.

        👉 Use the top menu to get started.
        """)

    elif selected == "About":
        st.markdown("""
        The Wind Power Forecasting AI app is a user-friendly, intelligent tool designed to help energy professionals predict active power output from wind turbines using machine learning techniques.

        Built on a powerful K-Nearest Neighbors (KNN) regression model, the app ingests real-time or historical turbine parameters such as wind speed, gearbox oil temperature, generator winding temperatures, and reactive power to accurately forecast performance.

        This forecasting platform empowers:
        - Energy companies to plan for power delivery and load balancing
        - Engineers to monitor turbine health and avoid overheating or underperformance
        - Analysts to evaluate performance trends and make data-driven decisions

        Key Functionalities:
        - Manual entry and CSV batch prediction modes
        - Smart, real-time suggestions to optimize turbine health
        - Secure user login and persistent prediction history

        Developed by:  
        E. Kuladeep  
        J. Yashwanth  
        G. Aquill Rao   

        
        """)    

    

    
        
