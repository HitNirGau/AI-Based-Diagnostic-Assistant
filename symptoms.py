import streamlit as st
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key='AIzaSyALXzGMeaViJQfJT9cx6KR7ac8Ef-uD5uM')

def get_response(prompt):
    model = genai.GenerativeModel("gemini-pro")  # Use Gemini-Pro model
    response = model.generate_content(f"The patient has {prompt}. Possible diseases are:")
    return response.text if response.text else "I'm not sure. Please provide more details."

st.title("🩺 Medical Chatbot")
st.write("Describe your symptoms, and I'll suggest possible diseases.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Enter your symptoms:", key="input")
if st.button("Send"):
    if user_input:
        response = get_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

st.write("Chat History")
for role, message in st.session_state.chat_history:
    st.markdown(f"**{role}:** {message}")

