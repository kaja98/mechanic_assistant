import streamlit as st
# from dotenv import load_dotenv
from main_old import run_pipeline

# load_dotenv()


st.set_page_config(page_title="Mechanic Assistant", page_icon="🔧")

st.title("🔧 Mechanic Assistant Chatbot")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your machine..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = run_pipeline(prompt)
            st.markdown(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
