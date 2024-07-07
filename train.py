import json
import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import bitsandbytes as bnb

# Custom Trainer Class for Mixed Precision
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Load and preprocess dataset
with open('dataset/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert dataset to Hugging Face Dataset object
dataset = Dataset.from_list(data)

# Load model and tokenizer
model_name = 'maum-ai/Llama-3-MAAL-8B-Instruct-v0.1'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples['chat'], padding="max_length", truncation=True, max_length=256)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define LoRA configuration
lora_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # Specify target modules explicitly if necessary
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, lora_config)

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Adjust based on VRAM and model size
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Use mixed precision
    save_steps=10_000,  # Save model every 10,000 steps
    save_total_limit=2,  # Keep the last 2 checkpoints
    remove_unused_columns=False,
)

# Mixed Precision Training setup
scaler = GradScaler()

# Use bitsandbytes Adam8bit optimizer
optimizer = bnb.optim.Adam8bit(peft_model.parameters(), lr=training_args.learning_rate)

# Initialize Trainer
trainer = CustomTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    optimizers=(optimizer, None)  # Pass optimizer, with no scheduler
)

for epoch in range(training_args.num_train_epochs):
    peft_model.train()
    for step, batch in enumerate(trainer.get_train_dataloader()):
        optimizer.zero_grad()

        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    trainer.evaluate()

# Save the fine-tuned model
peft_model.save_pretrained('./lora_finetuned_model')
tokenizer.save_pretrained('./lora_finetuned_model')

print("Model training complete and saved.")