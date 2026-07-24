import streamlit as st
import ollama
import time
from typing import List, Optional
from pydantic import BaseModel, Field


# --- SCHEMA FOR PATIENT DATA EXTRACTION ---
class PatientDataExtraction(BaseModel):
    # Match the exact phrasing you want the model to look for
    requested_metrics: List[str] = Field(
        description="List of metrics the user wants to predict, e.g., 'readmission rate' or 'length of stay'."
    )
    characteristics: List[str] = Field(
        description="List of patient characteristics provided, e.g., '100 pounds', 'heart problems'."
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Set ONLY if there is not enough information to identify metrics or characteristics.",
    )


st.title("MedAgent")

with st.sidebar:
    st.button("New Inquiry", on_click=lambda: st.session_state.messages.clear())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DISPLAY CHAT HISTORY AND HISTORICAL METRICS ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If this assistant message has a saved duration, render it right below the text
        if message["role"] == "assistant" and "elapsed_time" in message:
            st.caption(f"Response generated in {message['elapsed_time']:.2f} seconds")

# --- HANDLING NEW INPUT ---
if prompt := st.chat_input("What would you like me to predict?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    start_time = time.time()

    with st.chat_message("assistant"):
        instruction = (
            f'Analyze the following user input: "{prompt}".\n\n'
            "Extract the 'requested_metrics' and the provided patient 'characteristics'.\n"
            "If the user did not provide enough information to extract these, set the 'error_message' field to: "
            "'I'm sorry, I don't have enough information to provide a prediction.'"
        )

        try:
            request = ollama.chat(
                model="qwen2.5:3b",
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

        # 1. Show the answer immediately
        st.markdown(response)

        # 2. Calculate final duration
        elapsed_time = time.time() - start_time
        st.caption(f"Response generated in {elapsed_time:.2f} seconds")

        # 3. Save the answer AND the runtime metrics together into session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "elapsed_time": elapsed_time,  # <-- Saved permanently for the history loop
            }
        )
