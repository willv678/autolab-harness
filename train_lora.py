import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
TRAIN_FILE = "sft_train.jsonl"
VAL_FILE = "sft_val.jsonl"
OUTPUT_DIR = "./autolab_lora_adapter"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        dataset_text_field="messages",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[*] LoRA adapter saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()