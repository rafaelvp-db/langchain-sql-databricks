# Databricks notebook source
!accelerate config default

# COMMAND ----------

import torch
import logging
from models.mpt import InstructionTextGenerationPipeline

# COMMAND ----------

# DBTITLE 1,Test 1.1: Simple Querying
# Here we define a standard prompt, where we give some of the details on where to find the data for getting results
# We also explicitly define which catalog we're interested in, though we might skip that step later on if we're able
# to fetch UC data in realtime

spark.sql("use catalog cnh")
spark.sql("use schema gold")

def get_data_insights(
  question,
  data_details_instruction,
  temperature = 0.3,
  max_new_tokens = 1000,
):

  query_prompt = f"""
    You are a SQL expert. You are provided with the following information about tables in a database/schema:
    '{data_details_instruction}'
    With these details in mind, write a SQL query which is syntatically correct in order to answer the following question about data stored in the mentioned table: '{question}'.
    Return the resulting SQL query, without including any kind of explanation or comments.
  """

  model = InstructionTextGenerationPipeline()

  result_query = model(
    instruction = query_prompt,
    max_new_tokens = max_new_tokens,
    temperature = temperature
  )
  
  return result_query

# COMMAND ----------

# DBTITLE 1,Getting the Insights
data_details_instruction = """

Warranty claims are stored in the `cnh.gold.claim_merged` table. The DDL statement for this table is as follows:

`CREATE TABLE cnh.gold.claim_merged (claim_number string, FDP_code (string), translated (string))`

"""

question = "How many distinct claims are there?"

query = get_data_insights(
  question = question,
  data_details_instruction = data_details_instruction,
  temperature = 0.1
)

print(f"Generated query: {query}")

final_query = f"""
  use catalog cnh;
  use schema gold;
  {query}
"""

# COMMAND ----------

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

def run_query(query):

  from databricks import sql

  databricks_hostname = "e2-demo-field-eng.cloud.databricks.com"
  databricks_http_path = "/sql/1.0/warehouses/ead10bf07050390f"

  with sql.connect(
    server_hostname = databricks_hostname,
    http_path       = databricks_http_path,
    access_token    = databricks_token
  ) as connection:

    with connection.cursor() as cursor:
      cursor.execute(query)
      result = cursor.fetchall()

      for row in result:
        print(row)

# COMMAND ----------

run_query(query)

# COMMAND ----------

# DBTITLE 1,Getting metadata from UC
import requests
import json
from databricks_cli.unity_catalog.api import UnityCatalogApi
from databricks_cli.sdk import ApiClient

def prettify_metadata(
  metadata: dict,
  include_metadata = True
) -> str:
  result = f"Table {metadata['name']} contains the following columns: \n"
  for column in metadata["columns"]:
    column_metadata = json.loads(column["type_json"])
    column_description = f"""
      Column name: {column["name"]}
      Column type: {column["type_name"]}
    """
    if include_metadata:
      column_description += f"\nMetadata: {column_metadata['metadata']}"
    result = result + column_description

  return result

def get_table_info(
  catalog,
  schema,
  table = None,
  include_metadata = True
):

  databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().getOrElse(None)
  access_token = dbutils.secrets.get(scope = "langchain", key = "token")

  headers = {
    "Authentication": f"Bearer {access_token}"
  }

  if table:

    url = f"https://{databricks_url}/api/2.1/unity-catalog/tables/{catalog}.{schema}.{table}"

    response = requests.get(
      url = url,
      headers = headers
    )
  else:
    # Table was not specified, browse through tables in the catalog.
    url = f"https://{databricks_url}/api/2.1/unity-catalog/tables/"
    response = requests.get(
      url = url,
      headers = headers
    )

  metadata = response.json()
  return metadata

# COMMAND ----------

# DBTITLE 1,Test 1.2: Specifying table details with UC metadata
# TODO

# COMMAND ----------


