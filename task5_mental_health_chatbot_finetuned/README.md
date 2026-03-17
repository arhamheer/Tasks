# Task 5: Mental Health Support Chatbot (Fine-Tuned)

## Objective
Fine-tune a small language model on empathetic dialogue and serve a supportive chatbot UI.

## Files
- `train.py`: fine-tunes DistilGPT2 on a subset of EmpatheticDialogues
- `chatbot.py`: loads fine-tuned model (or base fallback) and generates responses
- `app.py`: Streamlit frontend with enhanced visual design
- `model/`: saved local model after training

## Run
```bash
python train.py
streamlit run app.py
```

## Notes
- Training can take time based on hardware.
- If no fine-tuned model is found, the app runs with base `distilgpt2`.

## Troubleshooting
If you see an error similar to:

`ImportError: cannot import name 'NP_SUPPORTED_MODULES' from 'torch._dynamo.utils'`

your environment likely has incompatible `torch` / `transformers` versions from a partial upgrade.

Reinstall pinned dependencies from the project root:

```powershell
\.venv\Scripts\python.exe -m pip install --upgrade pip
\.venv\Scripts\python.exe -m pip install --force-reinstall -r requirements.txt
```

Then run task 5 again:

```powershell
\.venv\Scripts\python.exe train.py
\.venv\Scripts\python.exe -m streamlit run app.py
```
