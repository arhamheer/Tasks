# AI/ML Internship Tasks Repository

This repository contains 6 tasks, each in a separate folder with:
- Python solution scripts
- A Streamlit frontend
- A Jupyter notebook
- A `data/` folder for dataset files

## Task Folders
1. `task1_iris_exploration`
2. `task2_stock_prediction`
3. `task3_heart_disease_prediction`
4. `task4_health_query_chatbot`
5. `task5_mental_health_chatbot_finetuned`
6. `task6_house_price_prediction`

## Setup
```bash
pip install -r requirements.txt
```

## Environment Variables
Create a `.env` file in the repository root (`d:\Data\Delete\tasks\.env`) for API-based tasks, for example:

```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-3.5-turbo
HF_API_KEY=your_huggingface_key
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

## Run Any Task
Use each folder's `README.md` for exact commands.

## Run All Tasks Automatically
Use the PowerShell helper script to verify all tasks and optionally launch all Streamlit apps.

```powershell
.\run_all_tasks.ps1
.\run_all_tasks.ps1 -LaunchApps
.\run_all_tasks.ps1 -RunTask5Training
```

## Notes
- Some datasets are downloaded automatically when running scripts/apps.
- Task 4 requires API keys for OpenAI or Hugging Face.
- Task 5 fine-tuning may take longer depending on hardware.
