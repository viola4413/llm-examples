from schema import Conversation, Message, ModelConfig
from litellm import completion
import streamlit as st


FRIENDLY_MAPPING = {
    "Snowflake Arctic": "snowflake/snowflake-arctic-instruct",
    "LLaMa 3 8B": "meta/meta-llama-3-8b-instruct",
    "Mistral 7B": "mistralai/mistral-7b-instruct-v0.2",
}
AVAILABLE_MODELS = [k for k in FRIENDLY_MAPPING.keys()]


def generate_stream(
    conversation: Conversation,
):
    messages = conversation.messages
    model_config: ModelConfig = conversation.model_config
    full_model_name = FRIENDLY_MAPPING[model_config.model]

    if model_config.system_prompt:
        system_msg = Message(role="system", content=model_config.system_prompt)
        messages = [system_msg]
        messages.extend(conversation.messages)

    response = completion(
        model=f"replicate/{full_model_name}",
        messages=[dict(m) for m in messages],
        api_key=st.secrets.REPLICATE_API_TOKEN,
        stream=True,
        temperature=model_config.temperature,
        top_p=model_config.top_p,
        max_tokens=model_config.max_new_tokens,
    )
    for part in response:
        yield part.choices[0].delta.content or ""
