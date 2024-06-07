# LLM Evaluation Demo

This app enables conversation with several LLMs under various configurations,
including access to the Streamlit docs (via RAG) and Automated Evaluations and Tracing with TruLens

Log in to save your conversations. Use Admin Mode to manage users.

**You can access the hosted application and Evaluation Dashboard at:**
[TruLens Evaluation Dashboard](https://llm.truera.net:8484/).
[Streamlit Chat App](http://llm.truera.net:8502)

## Goals

- Highlight many of the new features launched in Streamlit in the last ~6 months
- Show how those features can work together to deliver a functional, useful app
- Provide some templates for solving common AI application problems around evaluating LLMs.
- Show how to integrate RAGs in Streamlit to ground rags on relevant documents
- Demonstrate how observability via TruLens can drive rapid experimentation and improvement of LLM apps.

## Running the app

1. Clone the repo, create a new python virtual environment, and do `pip install -r requirements.txt`
2. You'll need a `REPLICATE_API_TOKEN` to call the models and a `PINECONE_API_KEY` to run retrieval. Add it to Streamlit secrets or ENV
   - You can also configure `enablePersistence = false` in Streamlit secrets to disable saving
     out changes to the `.jsonl` file.
3. Do `streamlit run app.py` to run the chat app locally.
4. Do `python launch_trulens_dashboard.py` to run the evaluation dashboard locally.

## Contributing

We are welcoming small improvements to this app as PRs and bigger ideas as Github issues.
Feel free to tag `@sfc-gh-jreini` for faster responses.