import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig, TaskType

# Model configuration
MODEL_NAME = "/workspace/models/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/"  # You can use llama-2 instead, or specify the Llama 3.2 variant
OUTPUT_DIR = "./llama-classification-model"

# Check GPU availability
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

# Load model and tokenizer
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,  # Adjust based on your classification task
    device_map="auto",
    local_files_only=True,
)

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load and preprocess dataset
print("Loading dataset...")
# Example: using a public dataset. Replace with your own dataset
dataset = load_dataset("imdb")  # Change to your dataset


def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )


processed_datasets = dataset.map(preprocess_function, batched=True)
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["test"]

# Initialize TensorBoard writer
tensorboard_dir = os.path.join(OUTPUT_DIR, "runs")
writer = SummaryWriter(log_dir=tensorboard_dir)
print(f"TensorBoard logs will be saved to: {tensorboard_dir}")

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-4,
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    bf16=torch.cuda.is_available(),
    logging_steps=100,
    gradient_accumulation_steps=4,
    report_to=["tensorboard"],
    logging_dir=tensorboard_dir,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# Train the model
print("Starting training...")
trainer.train()

# Close TensorBoard writer
writer.close()

print("Training completed!")
print(f"Model saved to: {OUTPUT_DIR}")
print(f"View TensorBoard logs with: tensorboard --logdir {tensorboard_dir}")
