# Databricks notebook source
# MAGIC %pip install langchain databricks-sql-connector bitsandbytes

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain import SQLDatabase, SQLDatabaseChain
from langchain.llms import HuggingFacePipeline

# COMMAND ----------

token = "TOKEN"
http_path = "/sql/1.0/warehouses/cf8e3ce76d2288d6"

db = SQLDatabase.from_uri(f"databricks://token:{token}@adb-984752964297111.11.azuredatabricks.net?http_path={http_path}&catalog=poc&schema=sensors")

# COMMAND ----------

import torch
from transformers import pipeline

model_name = "databricks/dolly-v2-3b"
instruct_pipeline = pipeline(
  model=model_name,
  torch_dtype = torch.bfloat16,
  trust_remote_code=True,
  device_map="auto",
  return_full_text=True,
  max_new_tokens=256,
  top_p=0.95,
  top_k=50,
  model_kwargs = {"max_position_embeddings": 5214}
)

hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)

# COMMAND ----------

db_chain = SQLDatabaseChain.from_llm(llm = hf_pipe, db = db, verbose=True)

# COMMAND ----------

db_chain.run("How many GOOD events in Bronze?")

# COMMAND ----------


