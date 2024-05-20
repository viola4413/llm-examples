from copy import deepcopy
from typing import List
from pathlib import Path
from schema import ConversationRecord


CURRENT_DIR = Path(__file__).parent.resolve()
CONVERSATION_HISTORY_FILE = CURRENT_DIR / "data" / "conversation_history.jsonl"


def load_chat_history() -> List[ConversationRecord]:
    chat_history = []
    with open(CONVERSATION_HISTORY_FILE, "r") as f:
        jl = f.readlines()
    for c in jl:
        chat_history.append(ConversationRecord.from_json(c))
    return chat_history


class ConversationManager:
    def __init__(self):
        self._conversations = load_chat_history()

    def get_by_title(self, title: str) -> ConversationRecord:
        index = [c.title for c in self._conversations].index(title)
        return deepcopy(self._conversations[index])

    def get_user_conversation_titles(self, user: str) -> List[str]:
        return [c.title for c in self._conversations if c.user == user]

    def add_or_update(self, conv: ConversationRecord, persist=False):
        conv_copy = deepcopy(conv)
        # Update existing record if exists
        try:
            index = [c.title for c in self._conversations].index(conv.title)
            self._conversations[index] = conv_copy
        # Else add a new one
        except ValueError:
            self._conversations.append(conv_copy)
        if persist:
            self.persist_records()

    def persist_records(self):
        jsonl = []
        for conv in self._conversations:
            jsonl.append(conv.to_json())
        with open(CONVERSATION_HISTORY_FILE, "w") as f:
            f.write("\n".join(jsonl))
