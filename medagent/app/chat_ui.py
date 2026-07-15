import streamlit as st

st.title("MedAgent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like me to predict?"):
    with st.chat_message("user"):
        st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = "This is a placeholder response from the model."
        st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
