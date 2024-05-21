import streamlit as st

from common_ui import page_setup

page_setup("About")

st.markdown(
    """
This app enables conversation with several LLMs under various configurations,
along with simple human feedback and persisted conversation history.

Log in to save your conversations. Use Admin Mode to manage users, view
aggregated feedback stats, as well as view automated evaluation _(coming soon!)_.
"""
)
