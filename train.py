import json
import torch
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from torch.cuda.amp import GradScaler
from torch.quantization import quantize_dynamic
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments

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

# 모델의 일부 파라미터 동결
for param in model.base_model.parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Mixed Precision Training 설정
scaler = GradScaler()

# Optimizer 설정 (bitsandbytes를 사용하여 8-bit optimizer 적용)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=training_args.learning_rate)

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer
)

# 훈련
trainer.train()

# 모델 양자화
model.eval()  # 모델을 평가 모드로 전환
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# 양자화된 모델 저장
quantized_model.save_pretrained('./quantized_model')

print("모델 훈련이 완료되었고, 양자화된 모델이 저장되었습니다.")