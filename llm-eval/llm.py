import replicate
from schema import Conversation, ModelConfig


def encode_arctic(conversation: Conversation):
    prompt = []
    for msg in conversation.messages:
        prompt.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")

    prompt.append("<|im_start|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    return prompt_str


def encode_llama3(conversation: Conversation):
    prompt = []
    prompt.append("<|begin_of_text|>")
    for msg in conversation.messages:
        prompt.append(f"<|start_header_id|>{msg.role}<|end_header_id|>")
        prompt.append(f"{msg.content.strip()}<|eot_id|>")
    prompt.append("<|start_header_id|>assistant<|end_header_id|>")
    prompt.append("")
    prompt_str = "\n\n".join(prompt)
    return prompt_str


def encode_generic(conversation: Conversation):
    prompt = []
    for msg in conversation.messages:
        if msg.role == "user":
            prompt.append("user:\n" + msg.content)
        else:
            prompt.append("assistant:\n" + msg.content)

    prompt.append("assistant:")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    return prompt_str


ENCODING_MAPPING = {
    "snowflake/snowflake-arctic-instruct": encode_arctic,
    "meta/meta-llama-3-8b": encode_llama3,
    "mistralai/mistral-7b-instruct-v0.2": encode_generic,
}


def generate_stream(
    conversation: Conversation,
    model_config: ModelConfig,
):
    prompt_str = ENCODING_MAPPING[model_config.model](conversation)

    model_input = {
        "prompt": prompt_str,
        "prompt_template": r"{prompt}",
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
    }
    stream = replicate.stream(model_config.model, input=model_input)

    for t in stream:
        yield str(t)
