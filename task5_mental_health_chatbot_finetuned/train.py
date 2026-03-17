from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_NAME = "distilgpt2"


def format_dialogue(example):
    prompt = (
        "You are an empathetic mental health support assistant. "
        "Respond warmly, gently, and avoid clinical diagnosis.\n"
        f"Situation: {example['prompt']}\n"
        f"User: {example['utterance']}\n"
        f"Assistant: {example['context']}"
    )
    return {"text": prompt}


def main() -> None:
    dataset = load_dataset("empathetic_dialogues", split="train[:2%]")

    prepared = dataset.map(format_dialogue)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    tokenized = prepared.map(tokenize, batched=True, remove_columns=prepared.column_names)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=25,
        save_steps=200,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"Saved fine-tuned model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
