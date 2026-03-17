from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
BASE_MODEL = "distilgpt2"


class MentalHealthSupportBot:
    def __init__(self) -> None:
        model_source = str(MODEL_DIR) if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()) else BASE_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_source)
        self.model.eval()

        self.model_source = model_source

    def build_prompt(self, user_text: str) -> str:
        return (
            "You are a calm and empathetic emotional wellness assistant. "
            "Validate feelings, suggest gentle coping steps, and encourage seeking trusted support when needed. "
            "Do not provide diagnosis or emergency instruction.\n"
            f"User: {user_text}\n"
            "Assistant:"
        )

    def generate(self, user_text: str, max_new_tokens: int = 80) -> str:
        prompt = self.build_prompt(user_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.92,
                temperature=0.8,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "Assistant:" in generated:
            return generated.split("Assistant:", 1)[1].strip()
        return generated.strip()
