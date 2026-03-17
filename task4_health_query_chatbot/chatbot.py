import os
from pathlib import Path
from typing import Dict

import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ROOT_ENV_PATH = BASE_DIR.parent / ".env"


def _load_env_fallback(env_path: Path) -> None:
    """Fallback parser for .env files saved with encodings like UTF-16."""
    if not env_path.exists():
        return

    raw = env_path.read_bytes()
    decoded = None
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            text = raw.decode(encoding)
            if "=" in text:
                decoded = text
                break
        except UnicodeDecodeError:
            continue

    if decoded is None:
        return

    for line in decoded.splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item.startswith("export "):
            item = item[len("export ") :].strip()
        if "=" not in item:
            continue

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load keys from workspace-level .env first, then fall back to default discovery.
load_dotenv(ROOT_ENV_PATH)
load_dotenv()
_load_env_fallback(ROOT_ENV_PATH)


SYSTEM_PROMPT = (
    "You are a helpful medical assistant for general health education only. "
    "Use simple, friendly language. Provide high-level information, not diagnosis. "
    "Always include a short safety note to consult a licensed doctor for personal advice."
)

BLOCKED_TOPICS = [
    "suicide",
    "self-harm",
    "kill myself",
    "overdose",
    "stop medication",
    "replace doctor",
]


class HealthChatbot:
    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_openai_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
        self.azure_openai_chat_api_version = os.getenv(
            "AZURE_OPENAI_CHAT_API_VERSION", "2024-10-21"
        )

        self.hf_api_key = os.getenv("HF_API_KEY", "")
        self.hf_model = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

    def get_provider(self) -> str:
        if self.openai_api_key:
            return "openai"

        if (
            self.azure_openai_api_key
            and self.azure_openai_endpoint
            and self.azure_openai_chat_deployment
        ):
            return "azure-openai"

        if self.hf_api_key:
            return "huggingface"

        return "none"

    def provider_readiness(self) -> Dict[str, bool]:
        """Expose non-sensitive readiness flags for UI and verification."""
        return {
            "openai": bool(self.openai_api_key),
            "azure_openai": bool(
                self.azure_openai_api_key
                and self.azure_openai_endpoint
                and self.azure_openai_chat_deployment
            ),
            "huggingface": bool(self.hf_api_key),
        }

    def is_blocked(self, query: str) -> bool:
        q = query.lower()
        return any(topic in q for topic in BLOCKED_TOPICS)

    def safety_response(self) -> str:
        return (
            "I cannot help with harmful or emergency medical instructions. "
            "Please contact emergency services or a qualified healthcare professional immediately."
        )

    def build_prompt(self, query: str) -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "User question: "
            f"{query}\n\n"
            "Assistant response format:\n"
            "1) Brief explanation\n"
            "2) Practical general tips\n"
            "3) Safety note\n"
        )

    def ask_openai(self, query: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    def ask_azure_openai(self, query: str) -> str:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=self.azure_openai_api_key,
            azure_endpoint=self.azure_openai_endpoint,
            api_version=self.azure_openai_chat_api_version,
        )

        response = client.chat.completions.create(
            model=self.azure_openai_chat_deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    def ask_huggingface(self, query: str) -> str:
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        payload: Dict[str, object] = {
            "inputs": self.build_prompt(query),
            "parameters": {"max_new_tokens": 220, "temperature": 0.5},
        }
        url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and data:
            generated = data[0].get("generated_text", "")
            return generated.split("Assistant response format:")[-1].strip() or generated.strip()
        return str(data)

    def ask(self, query: str) -> str:
        if self.is_blocked(query):
            return self.safety_response()

        provider = self.get_provider()

        try:
            if provider == "openai":
                return self.ask_openai(query)

            if provider == "azure-openai":
                return self.ask_azure_openai(query)

            if provider == "huggingface":
                return self.ask_huggingface(query)
        except Exception as exc:
            return f"API call failed using provider '{provider}': {exc}"

        return (
            "No API key found. Set OPENAI_API_KEY, AZURE_OPENAI_* or HF_API_KEY in environment variables. "
            "For now, this bot can only return this fallback message."
        )
