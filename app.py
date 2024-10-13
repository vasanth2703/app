# app.py (Streamlit frontend)

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Heart Health Dashboard", layout="wide")

# Define API URL
API_URL = "https://fastapi-app-yyxx.onrender.com"

# Initialize session state
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'is_signed_in' not in st.session_state:
    st.session_state.is_signed_in = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'echoMonitorData' not in st.session_state:
    st.session_state.echoMonitorData = []
if 'ai_responses' not in st.session_state:
    st.session_state.ai_responses = []

def sign_in(email, password):
    response = requests.post(
        f"{API_URL}/token",
        data={"username": email, "password": password}
    )
    if response.status_code == 200:
        token_data = response.json()
        st.session_state.access_token = token_data["access_token"]
        st.session_state.is_signed_in = True
        user_response = requests.get(
            f"{API_URL}/users/me",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
        )
        if user_response.status_code == 200:
            user_data = user_response.json()
            st.session_state.user_type = user_data["role"]
        return True
    return False

def sign_out():
    st.session_state.is_signed_in = False
    st.session_state.user_type = None
    st.session_state.access_token = None

def register_user(email, password, role):
    try:
        response = requests.post(
            f"{API_URL}/register",
            json={"email": email, "password": password, "role": role}
        )
        response.raise_for_status()  # Raises an HTTPError for bad responses
        st.success("Registration successful. Please sign in.")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Registration failed. Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response content: {e.response.text}")
        return False
    
def get_doctors():
    try:
        response = requests.get(
            f"{API_URL}/doctors",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get doctors: {str(e)}")
        return []




def get_network_posts():
    try:
        response = requests.get(
            f"{API_URL}/posts",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get posts: {str(e)}")
        return []

def create_post(content, post_type):
    try:
        response = requests.post(
            f"{API_URL}/posts",
            json={"content": content, "post_type": post_type},
            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
        )
        response.raise_for_status()
        st.success("Post created successfully")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create post: {str(e)}")


def analyze_file(file, file_type):
    try:
        files = {"file": file}
        response = requests.post(
            f"{API_URL}/analyze",
            files=files,
            data={"file_type": file_type},
            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to analyze file: {str(e)}")
        return {"error": "Failed to analyze file"}
    
def add_health_data(heart_rate):
    try:
        response = requests.post(
            f"{API_URL}/health-data",
            json={"heart_rate": heart_rate},
            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
        )
        response.raise_for_status()
        st.success(f"Heart rate {heart_rate} bpm submitted successfully")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to add health data: {str(e)}")

def get_health_data():
    try:
        response = requests.get(
            f"{API_URL}/health-data",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get health data: {str(e)}")
        return []

def display_health_monitor():
    st.header("Health Monitor")
    
    if st.session_state.user_type == "Client":
        # Display wearable device connection for clients
        st.subheader("Wearable Device Connection")
        if st.button("Connect to Wearable Device"):
            st.success("Connected to wearable device")
        
        # Display current heart rate
        current_heart_rate = st.number_input("Enter your current heart rate (bpm)", min_value=40, max_value=200, value=70)
        if st.button("Submit Heart Rate"):
            add_health_data(current_heart_rate)
            st.success(f"Heart rate {current_heart_rate} bpm submitted successfully")


        # Fetch and display health data history
        health_data =  [
            {"date": "2023-01-01", "heart_rate": 72},
            {"date": "2023-01-02", "heart_rate": 75},
            {"date": "2023-01-03", "heart_rate": 70},
        ]
        if health_data:
            df = pd.DataFrame(health_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['heart_rate'], mode='lines+markers', name='Heart Rate'))
            fig.update_layout(title='Heart Rate History', xaxis_title='Date', yaxis_title='Heart Rate (bpm)')
            st.plotly_chart(fig)
        else:
            st.write("No health data available.")
    
    elif st.session_state.user_type == "Doctor/Technician":
        # Display Echo Monitor for doctors/technicians
        st.subheader("Echo Monitor")
        if st.button("Connect to Echo Device"):
            st.success("Connected to echo device")
            # Simulate receiving echo data
            st.session_state.echoMonitorData = [
                {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "data": "Echo data 1"},
                {"timestamp": (datetime.now() + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"), "data": "Echo data 2"},
            ]
        
        # Display echo data
        if st.session_state.echoMonitorData:
            for echo_data in st.session_state.echoMonitorData:
                st.write(f"{echo_data['timestamp']}: {echo_data['data']}")
            
            # AI-generated real-time echo report
            if st.button("Generate Echo Report"):
                st.success("AI-generated Echo Report:")
                st.write("Patient shows normal cardiac function with no significant abnormalities detected.")
        else:
            st.write("No echo data available. Please connect to the echo device.")

def display_network():
    st.header("Heart Health Network")
    
    # Create a new post
    new_post = st.text_area("Share your thoughts or experience")
    post_type = st.selectbox("Post type", ["Text", "Article", "Video"])
    if st.button("Post"):
        create_post(new_post, post_type.lower())
        st.success("Post created successfully!")

    # Display existing posts
    posts = get_network_posts()
    for post in posts:
        st.subheader(f"{post['author']} - {post['type'].capitalize()}")
        st.write(post['content'])
        col1, col2 = st.columns(2)
        col1.write(f"❤️ {post['likes']} Likes")
        col2.write(f"💬 {post['comments']} Comments")
        st.write("---")

def display_file_upload():
    st.header("AI Analysis")
    
    file_type = st.selectbox("Select file type", ["Image", "Video", "Report"])
    uploaded_file = st.file_uploader(f"Upload {file_type}", type=["png", "jpg", "mp4", "pdf", "txt"])
    
    if uploaded_file is not None and st.button("Analyze"):
        analysis_result = analyze_file(uploaded_file, file_type.lower())
        if "error" not in analysis_result:
            st.session_state.ai_responses.append({
                "file_name": uploaded_file.name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_type": file_type,
                "response": analysis_result['report'],
                "verified_by": None
            })
            st.success("File analyzed successfully!")
        else:
            st.error(analysis_result['error'])
    
    # Display AI response history
    st.subheader("AI Response History")
    # Display AI response history
    st.subheader("AI Response History")
    for idx, response in enumerate(st.session_state.ai_responses):
        with st.expander(f"{response['file_name']} - {response['timestamp']}"):
            st.write(f"File Type: {response['file_type']}")
            st.write(response['response'])
            if response['verified_by']:
                st.write(f"✅ Verified by Dr. {response['verified_by']}")
            else:
                verify_doctor = st.selectbox("Select a doctor to verify", ["Dr. Smith", "Dr. Johnson", "Dr. Williams"], key=f"verify_doctor_{idx}")
                if st.button("Verify Response", key=f"verify_button_{idx}"):
                    response['verified_by'] = verify_doctor
                    st.success(f"Response verified by {verify_doctor}")
                    st.rerun()
def display_chat():
    st.header("Chat")
    
    for message in st.session_state.chat_messages:
        if message['sender'] == "You":
            st.text_input("You:", value=message['content'], key=f"msg_{message['timestamp']}", disabled=True)
        else:
            st.text_area(message['sender'], value=message['content'], key=f"msg_{message['timestamp']}", disabled=True)

    new_message = st.text_input("Type your message...")
    if st.button("Send"):
        if new_message:
            st.session_state.chat_messages.append({
                "sender": "You",
                "content": new_message,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
            # Here you would typically send the message to the backend and get a response
            # For now, we'll just simulate a response
            st.session_state.chat_messages.append({
                "sender": "Dr. Smith",
                "content": "Thank you for your message. I'll review it and get back to you soon.",
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
            st.rerun()

def main():
    st.title("All About Heart Health")
    
    if not st.session_state.is_signed_in:
        tab1, tab2 = st.tabs(["Sign In", "Register"])
        
        with tab1:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Sign In"):
                if sign_in(email, password):
                    st.success("Signed in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        with tab2:
            new_email = st.text_input("Email", key="new_email")
            new_password = st.text_input("Password", type="password", key="new_password")
            role = st.selectbox("Role", ["Client", "Doctor/Technician"])
            if st.button("Register"):
                if register_user(new_email, new_password, role):
                    st.success("Registration successful. Please sign in.")
                    st.rerun()
                else:
                    st.error("Registration failed. Please check the error message above and try again.")
    else:
        st.sidebar.title(f"Welcome, {st.session_state.user_type}")
        if st.sidebar.button("Sign Out"):
            sign_out()
            st.rerun()

        tab1, tab2, tab3, tab4 = st.tabs(["Health Monitor", "Network", "AI Analysis", "Chat"])

        with tab1:
            display_health_monitor()

        with tab2:
            display_network()

        with tab3:
            display_file_upload()

        with tab4:
            display_chat()

if __name__ == "__main__":
    main()
