# Databricks notebook source
# MAGIC %pip install trl
# MAGIC %pip install bitsandbytes
# MAGIC %pip install peft==0.3.0

# COMMAND ----------

# MAGIC %sql
# MAGIC USE tinycoder;

# COMMAND ----------

df = spark.sql("SELECT * FROM tinycoder_instruct").toPandas()
df['text'] = df["prompt"]+df["response"]
df.drop(columns=['prompt', 'response'], inplace=True)
display(df), df.shape

# COMMAND ----------

from datasets import load_dataset
from datasets import Dataset
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.05)

# COMMAND ----------


from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from transformers.trainer_callback import TrainerCallback
import os
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
import mlflow
     

# COMMAND ----------

mlflow.set_experiment(f"/Users/avinash.sooriyarachchi@databricks.com/codegen1b_prod")


# COMMAND ----------


target_modules = ['qkv_proj', 'out_proj','fc_in','fc_out','lm_head']
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules = target_modules,
    task_type="CAUSAL_LM",
)

base_dir = "/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/codegen21b_10epoch_V100/"

per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = 'adamw_hf'
learning_rate = 1e-5
max_grad_norm = 0.3
warmup_ratio = 0.03
lr_scheduler_type = "linear"

     

# COMMAND ----------


from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir=base_dir,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs = 10.0,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    # ddp_find_unused_parameters=False,

)

# COMMAND ----------

nf4_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16
)

# COMMAND ----------

model_id = 'Salesforce/codegen2-1B'


# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config, trust_remote_code=True)

# COMMAND ----------

model.config.use_cache = False

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
tokenizer.padding_side = 'right'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# COMMAND ----------

trainer = SFTTrainer(
    model,
    peft_config=lora_config,
    train_dataset=dataset['train'],
    eval_dataset = dataset['test'],
    dataset_text_field="text",
    max_seq_length=256,
    args=training_args,
)
#Upcast layer norms to float 32 for stability
for name, module in trainer.model.named_modules():
  if "norm" in name:
    module = module.to(torch.float32)

# COMMAND ----------

trainer.train()

# COMMAND ----------

# 1.19 for 1 epoch on v100
