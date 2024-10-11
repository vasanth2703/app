import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import io
import base64

 API_URL = "https://my-app-app-3opetubrnxatm4hggpjsjc.streamlit.app/"  # Adjust the domain as needed  # Update this if your API is hosted elsewhere

# Initialize session state
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'current_heart_rate' not in st.session_state:
    st.session_state.current_heart_rate = None
if 'ecg_data' not in st.session_state:
    st.session_state.ecg_data = None
if 'echo_monitor_data' not in st.session_state:
    st.session_state.echo_monitor_data = None

def login(email, password):
    try:
        # Make a POST request to obtain the token
        response = requests.post(f"{API_URL}/token", data={"username": email, "password": password})
        
        # Check if the response status code is 401 (Unauthorized)
        if response.status_code == 401:
            st.error("Invalid credentials. Please check your email and password.")
            return False
        
        # Raise any other HTTP error (e.g., 500 Server Error)
        response.raise_for_status()

        # If successful, extract the token
        token_data = response.json()
        st.session_state.token = token_data["access_token"]

        # Get the current user's information
        user_response = requests.get(f"{API_URL}/users/me", headers={"Authorization": f"Bearer {st.session_state.token}"})
        user_response.raise_for_status()
        
        # Extract user role and store it in session state
        user_data = user_response.json()
        st.session_state.user_role = user_data["role"]

        # Return True if login is successful
        return True

    # Handle connection errors (e.g., API server down or incorrect URL)
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the API server. Please check your connection or try again later.")
        return False

    # Handle other request-related exceptions
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {str(e)}")
        return False

def register(email, password, role):
    try:
        response = requests.post(
            f"{API_URL}/register",
            json={"email": email, "password": password, "role": role}
        )
        if response.status_code == 400:
            st.error("Email already registered. Please use a different email.")
            return False
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred during registration: {str(e)}")
        return False

def logout():
    st.session_state.token = None
    st.session_state.user_role = None
    st.session_state.connected = False
    st.session_state.current_heart_rate = None
    st.session_state.ecg_data = None
    st.session_state.echo_monitor_data = None

def get_doctors():
    response = requests.get(f"{API_URL}/doctors", headers={"Authorization": f"Bearer {st.session_state.token}"})
    if response.status_code == 200:
        return response.json()
    return []

def get_ai_responses():
    response = requests.get(f"{API_URL}/ai-responses", headers={"Authorization": f"Bearer {st.session_state.token}"})
    if response.status_code == 200:
        return response.json()
    return []

def verify_ai_response(response_id, doctor_id, comment):
    response = requests.post(
        f"{API_URL}/verify-ai-response",
        json={"response_id": response_id, "doctor_id": doctor_id, "comment": comment},
        headers={"Authorization": f"Bearer {st.session_state.token}"}
    )
    if response.status_code == 200:
        st.success("AI response verified successfully")
    else:
        st.error("Failed to verify AI response")

def get_chat_messages():
    response = requests.get(f"{API_URL}/chat-messages", headers={"Authorization": f"Bearer {st.session_state.token}"})
    if response.status_code == 200:
        return response.json()
    return []

def send_message(content):
    response = requests.post(
        f"{API_URL}/send-message",
        json={"content": content},
        headers={"Authorization": f"Bearer {st.session_state.token}"}
    )
    if response.status_code == 200:
        st.success("Message sent successfully")
    else:
        st.error("Failed to send message")

def get_health_data():
    response = requests.get(f"{API_URL}/health-data", headers={"Authorization": f"Bearer {st.session_state.token}"})
    if response.status_code == 200:
        return response.json()
    return []

def connect_device():
    st.session_state.connected = True
    if st.session_state.user_role == "client":
        st.session_state.current_heart_rate = 75  # Simulated heart rate
        st.session_state.ecg_data = "Normal sinus rhythm detected"
    else:
        st.session_state.echo_monitor_data = "Real-time echo data: Normal left ventricular function, no valvular abnormalities detected."

def analyze_ecg(file):
    content = file.getvalue()
    ecg_data = np.genfromtxt(io.StringIO(content.decode('utf-8')), delimiter=',')
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        ecg_data = pd.read_csv(io.StringIO(content.decode('utf-8')), header=None).values
    else:  # Assume it's a text file with comma-separated values
        ecg_data = np.genfromtxt(io.StringIO(content.decode('utf-8')), delimiter=',')
        
    response = requests.post(
        f"{API_URL}/analyze-ecg",
        json={"data": ecg_data.tolist()},
        headers={"Authorization": f"Bearer {st.session_state.token}"}
    )
    
    if response.status_code == 200:
        result = response.json()
        st.success("ECG analyzed successfully")
        st.write(f"Prediction: {result['prediction']}")
        st.write(f"Confidence: {result['confidence']:.2f}")
        st.text(result['report'])
    else:
        st.error("Failed to analyze ECG")

def main():
    st.set_page_config(page_title="Heart Health Dashboard", layout="wide")
    st.title("Heart Health Dashboard")

    if not st.session_state.token:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                if login(email, password):
                    st.success("Logged in successfully!")
                    st.rerun()
        
        with tab2:
            st.subheader("Register")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_role = st.radio("User Type", ["client", "doctor"])
            if st.button("Register"):
                if register(reg_email, reg_password, reg_role):
                    st.success("Registered successfully! You can now log in.")
    else:
        if st.sidebar.button("Logout"):
            logout()
            st.rerun()

        # Alert
        st.warning(
            "Alert: " + 
            ("Your heart rate is elevated." if st.session_state.user_role == "client" else "Patient alert: Elevated heart rate detected.")
        )

        # Tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(["Health Monitor", "Doctor Network", "File Upload", "Chat"])

        # Health Monitor Tab
        with tab1:
            st.header("Health Monitor")
            if st.button("Connect Device"):
                connect_device()
            
            if st.session_state.connected:
                st.success("Device Connected")
                if st.session_state.user_role == "client":
                    st.metric("Current Heart Rate", f"{st.session_state.current_heart_rate} bpm")
                    st.write(f"ECG Analysis: {st.session_state.ecg_data}")
                    
                    # Fetch and display heart rate data
                    health_data = get_health_data()
                    if health_data:
                        df = pd.DataFrame(health_data)
                        df['date'] = pd.to_datetime(df['date'])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df['date'], y=df['heart_rate'], mode='lines'))
                        fig.update_layout(title="Heart Rate Over Time",
                                          xaxis_title="Date",
                                          yaxis_title="Heart Rate (bpm)")
                        st.plotly_chart(fig)
                else:
                    st.write("Echo Monitor Data:")
                    st.write(st.session_state.echo_monitor_data)
            else:
                st.warning("Please connect your device")

        # Doctor Network Tab
        with tab2:
            st.header("Doctor Network")
            doctors = get_doctors()
            search_query = st.text_input("Search doctors or specialties")
            filtered_doctors = [d for d in doctors if search_query.lower() in d['name'].lower() or search_query.lower() in d['specialty'].lower()]
            
            for doctor in filtered_doctors:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"{doctor['name']} ({doctor['id']})")
                    st.write(f"Specialty: {doctor['specialty']}")
                    st.write(f"Rating: {doctor['rating']}")
                    if doctor.get('distance'):
                        st.write(f"Distance: {doctor['distance']}")
                with col2:
                    if st.button(f"Book Appointment", key=doctor['id']):
                        st.success(f"Appointment request sent to {doctor['name']}")

        # File Upload Tab
        with tab3:

            st.header("File Upload")
            file_type = st.selectbox("File Type", ["echo", "ecg", "xray", "image", "video", "report"])
            
            # Define file types based on the selected file type
            if file_type == "ecg":
                allowed_types = ["csv", "txt"]
            elif file_type in ["image", "xray"]:
                allowed_types = ["jpg", "jpeg", "png"]
            elif file_type == "video":
                allowed_types = ["mp4", "avi", "mov"]
            else:  # For echo and report
                allowed_types = ["pdf", "dcm"]
            
            uploaded_file = st.file_uploader(f"Upload {file_type}", type=allowed_types, key=f"uploader_{file_type}")
            
            if uploaded_file:
                if file_type == "ecg":
                    analyze_ecg(uploaded_file)
                else:
                    file_contents = uploaded_file.read()
                    encoded_file = base64.b64encode(file_contents).decode()
                    
                    response = requests.post(
                        f"{API_URL}/analyze-file",
                        json={"file_type": file_type, "file_name": uploaded_file.name, "file_content": encoded_file},
                        headers={"Authorization": f"Bearer {st.session_state.token}"}
                    )
                    
                    if response.status_code == 200:
                        ai_response = response.json()
                        st.success("File uploaded and analyzed successfully")
                        st.write(f"AI Response: {ai_response['content']}")
                    else:
                        st.error("Failed to analyze file. Our team is working on implementing this functionality. Please check back later.")

            st.subheader("AI Response History")
            ai_responses = get_ai_responses()
            for response in ai_responses:
                with st.expander(f"{response['fileName']} - {response['timestamp']}"):
                    st.write(response['content'])
                    if st.session_state.user_role == "doctor" and not response['verifiedBy']:
                        doctor_id = st.selectbox("Select verifying doctor", [d['id'] for d in doctors], key=f"doctor_select_{response['id']}")
                        comment = st.text_input("Verification comment", key=f"comment_{response['id']}")
                        if st.button("Verify", key=f"verify_{response['id']}"):
                            verify_ai_response(response['id'], doctor_id, comment)
                    elif response['verifiedBy']:
                        st.info(f"Verified by: {response['verifiedBy']}")

        # Chat Tab
        with tab4:
            st.header("Chat")
            messages = get_chat_messages()
            for message in messages:
                if message['sender'] == "You":
                    st.text_input("You", value=message['content'], key=f"msg_{message['id']}", disabled=True)
                else:
                    st.text_area(message['sender'], value=message['content'], key=f"msg_{message['id']}", disabled=True)
            
            new_message = st.text_input("Type your message")
            if st.button("Send"):
                send_message(new_message)
                st.experimental_rerun()

if __name__ == "__main__":
    main()

