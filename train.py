import json
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
from huggingface_hub import DataCollatorForLanguageModeling
from datasets import Dataset


with open('dataset/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

model_name = 'maum-ai/Llama-3-MAAL-8B-Instruct-v0.1'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['chat'], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

trainer.train()