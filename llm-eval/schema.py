from typing import List, Literal
from pydantic import BaseModel
import streamlit as st


class ModelConfig(BaseModel):
    model: str
    temperature: float
    top_p: float
    max_new_tokens: int


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Conversation:
    messages: List[Message] = []

    def __init__(self):
        self.reset_messages()

    def add_message(self, message: Message, container=None, render=True):
        self.messages.append(message)
        if render:
            self.render_message(message, container)

    def reset_messages(self):
        self.messages = []

    def render_all(self, container=None):
        for message in self.messages:
            self.render_message(message, container)

    def render_message(self, message: Message, container=None):
        if container is not None:
            container.chat_message(message.role).write(message.content)
        else:
            st.chat_message(message.role).write(message.content)
