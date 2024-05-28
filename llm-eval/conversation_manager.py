from copy import deepcopy
from typing import Mapping, List, Set, Tuple
from pathlib import Path
from schema import Conversation, ConversationRecord


CURRENT_DIR = Path(__file__).parent.resolve()
CONVERSATION_HISTORY_FILE = CURRENT_DIR / "data" / "conversation_history.jsonl"


def load_chat_history() -> Mapping[str, ConversationRecord]:
    chat_history = dict()
    with open(CONVERSATION_HISTORY_FILE, "r") as f:
        jl = f.readlines()
    for c in jl:
        cr = ConversationRecord.from_json(c)
        chat_history[cr.id] = cr
    return chat_history


class ConversationManager:
    def __init__(self):
        self._conversations = load_chat_history()

    def list_users(self) -> Set[str]:
        return set([c.user for c in self._conversations.values()])

    def get_by_id(self, id: str) -> ConversationRecord:
        cr = self._conversations[id]
        return deepcopy(cr)

    def list_conversations_by_user(self, user: str) -> List[Tuple[str, str]]:
        return [(c.id, c.title) for c in self._conversations.values() if c.user == user]

    def get_all_conversations(self, *, user: str = None) -> List[Conversation]:
        all_records = self._conversations.values()
        if user:
            all_records = [c for c in self._conversations.values() if c.user == user]
        all_conversations = []
        for rec in all_records:
            all_conversations.extend(rec.conversations)
        return deepcopy(all_conversations)

    def add_or_update(self, conv: ConversationRecord, persist=False):
        conv_copy = deepcopy(conv)
        # Update existing record if exists or add a new one
        self._conversations[conv_copy.id] = conv_copy
        if persist:
            self.persist_records()

    def persist_records(self):
        jsonl = []
        for conv in self._conversations.values():
            jsonl.append(conv.to_json())
        with open(CONVERSATION_HISTORY_FILE, "w") as f:
            f.write("\n".join(jsonl))
