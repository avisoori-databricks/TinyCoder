# Databricks notebook source
# MAGIC %pip install bitsandbytes
# MAGIC %pip install accelerate
# MAGIC %pip install peft==0.3.0

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

# COMMAND ----------

# Replace the path names based on where the adapters, model and tokenizers were saved during your work

# COMMAND ----------

peft_model_id = "/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/tinycoder"

# COMMAND ----------

pretrained_model_id = 'Salesforce/codegen2-1B'

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_id, device_map='auto', trust_remote_code=True
)

# COMMAND ----------

from peft import PeftModel, PeftConfig
config = PeftConfig.from_pretrained(peft_model_id)
config.base_model_name_or_path

# COMMAND ----------

peft_model = PeftModel.from_pretrained(model, peft_model_id)

# COMMAND ----------

merged_model_path = "/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/tinycoder/merged_model"

# COMMAND ----------

merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(merged_model_path)

# COMMAND ----------

merged_model_tokenizer_path = "/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/tinycoder/merged_model_tokenizer"

# COMMAND ----------

tokenizer.save_pretrained(merged_model_tokenizer_path)

# COMMAND ----------

#Ensure that the final model can be loaded from the saved path
model = AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")

# COMMAND ----------

model.eval()

# COMMAND ----------

#Ensure that the tokenizer can be loaded from the saved path
tokenizer = AutoTokenizer.from_pretrained(merged_model_tokenizer_path)

# COMMAND ----------

tokenizer.padding_side = 'right'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing preprocessing and prediction functions before composing the pyfunc

# COMMAND ----------

def build_prompt(instruction):
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  {}

  ### Response:
  """.format(instruction)
    return prompt

# COMMAND ----------

def parse(text):
    start_marker = '### Response:'
    end_marker = '### End'
    start_index = text.find(start_marker)
    end_index = text.find(end_marker, start_index + len(start_marker))
    
    return (text[start_index + len(start_marker):].strip() if start_index != -1 and end_index == -1
            else text[start_index + len(start_marker):end_index].strip() if start_index != -1
            else None)

# COMMAND ----------

def extract_response(text):
    start_marker = '### Response:'
    end_marker = '### End'
    start_index = text.find(start_marker)
    end_index = text.find(end_marker, start_index + len(start_marker))
    
    return (text[start_index + len(start_marker):].strip() if start_index != -1 and end_index == -1
            else text[start_index + len(start_marker):end_index].strip() if start_index != -1
            else None)

# COMMAND ----------

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Give me a python function square a list of numbers

### Response:
"""
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

# generation_output = model.generate(
#     input_ids=input_ids, max_new_tokens=128, penalty_alpha=0.5, top_k=4, 
# )
# response = extract_response(tokenizer.decode(generation_output[0]))
# print(response)

# COMMAND ----------

payload_pd = pd.DataFrame([[prompt]],columns=['text'])
payload_pd

# COMMAND ----------

input_example = payload_pd

# COMMAND ----------

def predict(model_input):
    import json
    question = model_input.iloc[:,0].to_list()[0] # get the first column
    prompt = build_prompt(question)
    input_ids = tokenizer(prompt, return_tensors="pt").to('cuda')
    generation_output = model.generate(
    input_ids=input_ids["input_ids"], max_new_tokens=75)
    output = parse(tokenizer.decode(generation_output[0]))
    result = {'response': output}
    return json.dumps(result)

# COMMAND ----------

predict(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log with MLFlow and Deploy

# COMMAND ----------

artifacts = {
"tokenizer_path": merged_model_tokenizer_path,
"model_path": merged_model_path,
}

# COMMAND ----------

import mlflow.pyfunc

class Tinycoder(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['tokenizer_path'])
    self.model = AutoModelForCausalLM.from_pretrained(context.artifacts['model_path'], torch_dtype=torch.bfloat16, trust_remote_code=True)
    self.model.to(device = "cuda")
    self.model.eval()

  def build_prompt(self, instruction):
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  {}

  ### Response:
  """.format(instruction)
    return prompt

  def parse(self, text):
    start_marker = '### Response:'
    end_marker = '### End'
    start_index = text.find(start_marker)
    end_index = text.find(end_marker, start_index + len(start_marker))
    
    return (text[start_index + len(start_marker):].strip() if start_index != -1 and end_index == -1
            else text[start_index + len(start_marker):end_index].strip() if start_index != -1
            else None)


  def predict(self, context, model_input):
    import json
    question = model_input.iloc[:,0].to_list()[0] # get the first column
    prompt = self.build_prompt(question)
    input_ids = self.tokenizer(prompt, return_tensors="pt").to('cuda')
    generation_output = self.model.generate(
    input_ids=input_ids["input_ids"], max_new_tokens=75)
    output = self.parse(self.tokenizer.decode(generation_output[0]))
    result = {'response': output}
    return json.dumps(result)

# COMMAND ----------

from sys import version_info
 
PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

# COMMAND ----------

import cloudpickle
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'transformers==4.28.1',
          "datasets==2.12.0",
          "accelerate==0.21.0",
          "bitsandbytes==0.40.0",
          'pandas',
          "sentencepiece",
          "py7zr",
          'cloudpickle=={}'.format(cloudpickle.__version__),
          'torch'],
      },
    ],
    'name': 'tinycoder_environment'
}

mlflow_pyfunc_model_path = "tinycoder_prod"

# COMMAND ----------

mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, python_model=Tinycoder(),artifacts=artifacts, conda_env=conda_env, input_example = input_example)

# COMMAND ----------


