from typing import get_args
from schema import SUPPORTED_METRICS, SUPPORTED_CHARACTERISTICS

# Introspect the allowed types automatically
allowed_metrics_list = get_args(SUPPORTED_METRICS)
allowed_chars_list = get_args(SUPPORTED_CHARACTERISTICS)


def generate_extraction_prompt(user_input: str) -> str:
    """Generates a dynamic prompt mapping semantic intents to our strict schema tokens."""
    metrics_str = "\n".join([f"- {m}" for m in allowed_metrics_list])
    chars_str = "\n".join([f"- {c}" for c in allowed_chars_list])

    instruction = (
        f"Allowed metrics:\n{metrics_str}\n"
        f"Allowed characteristics:\n{chars_str}\n\n"
        f'Analyze the following user input: "{user_input}".\n\n'
        "Extract the 'requested_metrics' and the provided patient 'characteristics'.\n"
        "Only include matching items from the allowed lists into the schema. Ignore unsupported items. "
        "If no allowed data can be extracted, set 'error_message' to: "
        "'I'm sorry, I don't have enough information to provide a prediction.'"
    )
    return instruction
