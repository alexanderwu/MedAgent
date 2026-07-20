import streamlit as st
from google import genai
from dotenv import load_dotenv

load_dotenv()

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
        client = genai.Client()

        request = client.interactions.create(
            model="gemini-3.5-flash",
            input='"'
            + prompt
            + '"'
            + " Please extract the requested prediction metric as well as provided characterisitics of the patient. Please provide the output in a the following format: [<prediction1>, <prediction2>, ...], [<characteristic1>, <characteristic2>, ...]. If you do not have enough information, say 'I'm sorry, I don't have enough information to provide a prediction.'",
        )

        if not request.output_text:
            response = "Gemini Error: No output text received from the model."
        elif (
            request.output_text
            == "I'm sorry, I don't have enough information to provide a prediction."
        ):
            response = request.output_text
        else:
            response = (
                "You gave the characteristics: "
                + request.output_text.split("], [")[1].replace("]", "").replace("[", "")
                + ". The model predicts: "
                + request.output_text.split("], [")[0].replace("]", "").replace("[", "")
                + "."
            )
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
