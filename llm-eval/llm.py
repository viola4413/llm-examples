from typing import AsyncIterator, List
import replicate
from schema import Conversation, Message
from retrieve import PineconeRetriever

import streamlit as st
import re

# replicate key for running model
from trulens_eval.tru_custom_app import instrument


FRIENDLY_MAPPING = {
    "Snowflake Arctic": "snowflake/snowflake-arctic-instruct",
    "LLaMa 3 8B": "meta/meta-llama-3-8b",
    "Mistral 7B": "mistralai/mistral-7b-instruct-v0.2",
}
AVAILABLE_MODELS = [k for k in FRIENDLY_MAPPING.keys()]

def encode_arctic(messages: List[Message]):
    prompt = []
    for msg in messages:
        prompt.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")

    prompt.append("<|im_start|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    return prompt_str


def encode_llama3(messages: List[Message]):
    prompt = []
    prompt.append("<|begin_of_text|>")
    for msg in messages:
        prompt.append(f"<|start_header_id|>{msg.role}<|end_header_id|>")
        prompt.append(f"{msg.content.strip()}<|eot_id|>")
    prompt.append("<|start_header_id|>assistant<|end_header_id|>")
    prompt.append("")
    prompt_str = "\n\n".join(prompt)
    return prompt_str


def encode_generic(messages: List[Message]):
    prompt = []
    for msg in messages:
        prompt.append(f"{msg.role}:\n" + msg.content)

    prompt.append("assistant:")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    return prompt_str


ENCODING_MAPPING = {
    "snowflake/snowflake-arctic-instruct": encode_arctic,
    "meta/meta-llama-3-8b": encode_llama3,
    "mistralai/mistral-7b-instruct-v0.2": encode_generic,
}
class StreamGenerator:
    retriever = PineconeRetriever()

    def get_last_user_message(self, prompt_str):
        # Regex to find the last 'user' message
        match = re.findall(r'<\|im_start\|>user\n(.*?)(?=<\|im_end\|>)', prompt_str, re.DOTALL)
        if match:
            return match[-1].strip()
        return ""
    
    def prepare_prompt(self, conversation: Conversation):
        messages = conversation.messages
        model_config = conversation.model_config
        full_model_name = FRIENDLY_MAPPING[model_config.model]

        if model_config.system_prompt:
            system_msg = Message(role="system", content=model_config.system_prompt)
            messages = [system_msg] + messages

        prompt_str = ENCODING_MAPPING[full_model_name](messages)

        # Extract the last user message from the prompt string
        message = None
        for message in messages[::-1]:
            if message.role == "user":
                break
        last_user_message = message.content if message and message.role == "user" else ""
        return last_user_message, prompt_str
    
    def _generate_stream_with_replicate(self, model_name: str, model_input: dict):
        stream_iter: AsyncIterator = replicate.stream(model_name, input=model_input)
        for t in stream_iter:
            yield str(t)
    
    def _write_stream_to_st(self, st_container, stream_iter: AsyncIterator) -> None:
        full_text_response = ""
        
        if st_container is None:
            full_text_response = st.write_stream(stream_iter)
        else:
            full_text_response = st_container.write_stream(stream_iter) 
            
        return full_text_response
    
    @instrument
    def generate_response(self, last_user_message: str, prompt_str: str, conversation: Conversation, st_container=None) -> str:
        model_config = conversation.model_config
        full_model_name = FRIENDLY_MAPPING[model_config.model]

        model_input = {
            "prompt": prompt_str,
            "prompt_template": r"{prompt}",
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
        }

        final_response = ""
        stream_iter = self._generate_stream_with_replicate(full_model_name, model_input)
        
        final_response = self._write_stream_to_st(st_container, stream_iter) 

        return final_response
 

    @instrument
    def retrieve_context(self, last_user_message: str, prompt_str: str, conversation: Conversation):
        nodes = self.retriever.retrieve(query=last_user_message)
        context_message = "\n\n".join([node.get_content() for node in nodes])
        return context_message, nodes

    @instrument
    def retrieve_and_generate_response(self, last_user_message: str, prompt_str: str, conversation: Conversation, st_container=None) -> str:
        context_message, nodes = self.retrieve_context(last_user_message = last_user_message, prompt_str = prompt_str, conversation = conversation)  # Fixed by passing the conversation object instead of prompt_str
        model_config = conversation.model_config
        full_model_name = FRIENDLY_MAPPING[model_config.model]

        full_prompt = (
            "We have provided context information below. \n"
            "---------------------\n"
            f"{context_message}"
            "\n---------------------\n"
            f"Given this information, please answer the question: {prompt_str}"
        )

        model_input = {
            "prompt": full_prompt,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
        }
     
        final_response = ""
        stream_iter = self._generate_stream_with_replicate(full_model_name, model_input)

        final_response = self._write_stream_to_st(st_container, stream_iter) 
        
    
        return final_response
