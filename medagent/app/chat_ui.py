import streamlit as st
import pandas as pd

st.title("MedAgent")

with st.sidebar:
    st.button("New Inquiry", on_click=lambda: st.session_state.messages.clear())

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
        predicions_matrix = pd.DataFrame(
            {
                "Patient 0": [85, 0.01, 1],
            },
            index=["Length of Stay", "Readmission Rate", "Mortality Rate"],
        )
        st.write(predicions_matrix)

        st.session_state.messages.append({"role": "assistant", "content": response})
