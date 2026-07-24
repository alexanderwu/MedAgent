import streamlit as st
import ollama
import time

from schema import PatientDataExtraction
from prompts import generate_extraction_prompt

st.title("MedAgent")

with st.sidebar:
    st.button("New Inquiry", on_click=lambda: st.session_state.messages.clear())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "elapsed_time" in message:
            st.caption(f"Response generated in {message['elapsed_time']:.2f} seconds")

# --- HANDLING NEW INPUT ---
if prompt := st.chat_input("What would you like me to predict?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    start_time = time.time()

    with st.chat_message("assistant"):
        instruction = generate_extraction_prompt(prompt)
        try:
            request = ollama.chat(
                model="qwen2.5:7b",
                messages=[{"role": "user", "content": instruction}],
                format=PatientDataExtraction.model_json_schema(),
            )

            result = PatientDataExtraction.model_validate_json(
                request["message"]["content"]
            )

            if result.error_message:
                response = result.error_message
            else:
                chars_str = (
                    ", ".join(result.characteristics)
                    if result.characteristics
                    else "None specified"
                )
                preds_str = (
                    ", ".join(result.requested_metrics)
                    if result.requested_metrics
                    else "None specified"
                )
                response = f"You gave the characteristics: {chars_str}. The model predicts: {preds_str}."

        except Exception as exc:
            response = f"Ollama Error: {exc}"

        st.markdown(response)

        elapsed_time = time.time() - start_time
        st.caption(f"Response generated in {elapsed_time:.2f} seconds")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "elapsed_time": elapsed_time,
            }
        )
