# Databricks notebook source
# MAGIC %md
# MAGIC ### TinyCoder in Action: Putting it all together

# COMMAND ----------

# MAGIC %pip install solara

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

# COMMAND ----------

os.environ['DATABRICKS_TOKEN'] = "<token>"

# COMMAND ----------

def build_prompt(instruction):
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  {}

  ### Response:
  """.format(instruction)
    return prompt

# COMMAND ----------




def generatecode(txt):
  dataset = pd.DataFrame([[txt]],columns=['text'])
  url = 'https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/TinyCoder_1B_Instruct/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 
'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')}
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return response.json()

# COMMAND ----------

def metadatasearch(question):
  txt = question
  dataset = pd.DataFrame([[txt]],columns=['text'])
  url = 'https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/metadata_vectorsearch/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} 
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()


# COMMAND ----------

def predfunction(query):
  if 'sql' in query.lower():
    tableschema = json.loads(metadatasearch(query)['predictions'])['Response']['schema']
    txt_input =  query+ ", "+ tableschema
  else:
    txt_input = query

  prompt = build_prompt(txt_input)
  return json.loads(generatecode(txt_input)['predictions'])['response']
    



# COMMAND ----------

result = predfunction("Write a python function to multiply two lists")
print(result)

# COMMAND ----------

import solara

clicks = solara.reactive(' ')
text = solara.reactive("Define a python function to square a list of number")


@solara.component
def Page():
    solara.Info(f"TinyCoder-1B-Instruct: Demonstration of instruction following ability for Code Synthesis", color='#3640ff', icon=False)
    solara.InputText("Enter your instruction", value=text)
    color = "blue"
    # if clicks.value >= 5:
    #     color = "red"

    def increment():
        clicks.value = predfunction(text.value)
        print("clicks", clicks)

    solara.Button(label="Ask TinyCoder", on_click=increment, color=color)
    solara.Info(f"Response: {clicks}", color='black', icon=False)
Page()

# COMMAND ----------


