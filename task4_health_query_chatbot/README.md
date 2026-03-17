# Task 4: General Health Query Chatbot

## Objective
Answer general health questions using an LLM with prompt engineering and safety filters.

## Features
- Prompt style: helpful medical assistant tone
- Safety filter for harmful requests
- Support for either OpenAI API or Hugging Face Inference API

## Environment Variables
- `OPENAI_API_KEY` (optional)
- `OPENAI_MODEL` (default: `gpt-3.5-turbo`)
- `AZURE_OPENAI_API_KEY` (optional)
- `AZURE_OPENAI_ENDPOINT` (optional)
- `AZURE_OPENAI_CHAT_DEPLOYMENT` (optional)
- `AZURE_OPENAI_CHAT_API_VERSION` (default: `2024-10-21`)
- `HF_API_KEY` (optional)
- `HF_MODEL` (default: `mistralai/Mistral-7B-Instruct-v0.2`)

You can place these values in the root `.env` file (`d:\Data\Delete\tasks\.env`).
The chatbot loads this file automatically.

Provider priority: `OPENAI_API_KEY` -> Azure OpenAI vars -> `HF_API_KEY`.

## Run
```bash
python run_cli.py
streamlit run app.py
```
