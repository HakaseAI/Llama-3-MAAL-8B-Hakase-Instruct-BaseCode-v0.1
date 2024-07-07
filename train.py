import json
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
import torch
from datasets import Dataset
from torch.cuda.amp import autocast, GradScaler

# 데이터셋 로드
with open('dataset/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터셋을 Hugging Face의 Dataset 객체로 변환
dataset = Dataset.from_list(data)

# 모델과 토크나이저 로드
model_name = 'maum-ai/Llama-3-MAAL-8B-Instruct-v0.1'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# 토큰화 함수 정의
def tokenize_function(examples):
    return tokenizer(examples['chat'], padding="max_length", truncation=True)

# 데이터셋 토큰화
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 배치 크기를 줄이고, Gradient Accumulation 사용
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # 배치 크기를 줄임
    gradient_accumulation_steps=8,  # Gradient Accumulation 사용
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # 16-bit 부동 소수점 연산 사용
    save_steps=10_000,  # 모델 저장 빈도
    save_total_limit=2,  # 최대 저장 체크포인트 수
    remove_unused_columns=False,  # 데이터셋에 존재하지 않는 열 제거
    push_to_hub=False,  # 모델을 Hugging Face Hub에 푸시하지 않음
)

# 트레이너 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Mixed Precision Training 설정
scaler = GradScaler()

# Optimizer 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

# 사용자 정의 훈련 루프
for epoch in range(training_args.num_train_epochs):
    model.train()
    for step, batch in enumerate(trainer.get_train_dataloader()):
        optimizer.zero_grad()

        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    trainer.evaluate()

print("모델 훈련이 완료되었습니다.")
