# Databricks notebook source
from llms.instruction import InstructionLLM

# COMMAND ----------

llm = InstructionLLM()

# COMMAND ----------

# DBTITLE 1,Test 1.1: Simple Querying
# Here we define a standard prompt, where we give some of the details on where to find the data for getting results
# We also explicitly define which catalog we're interested in, though we might skip that step later on if we're able
# to fetch UC data in realtime

DATA_DETAILS_INSTRUCTION = """
    We have two tables, bronze and silver, and they are both part of the sensors schema.

    In the bronze table, we have the following columns:
      TagName (string): specifies the sensor ID
      EventTime (timestamp): timestamp during which the event occurred
      Status (string): Can be either 'Good' or 'Bad'
      Value (float): Contains the sensor reading for that particular point in time
  """

spark.sql("use catalog poc")
spark.sql("use schema sensors")

def get_data_insights(
  question,
  temperature = 0,
  max_new_tokens = 1000,
  data_details_instruction = DATA_DETAILS_INSTRUCTION
):

  query_prompt = f"""
    You are a SQL expert. You are provided with the following information about tables in a database/schema:
    '{data_details_instruction}'
    With these details in mind, write a SQL query which is syntatically correct in order to answer the following question: '{question}'.
    Return the resulting SQL query, without including any kind of explanation or comments.
  """

  result_query = llm(
    query_prompt,
    max_new_tokens = max_new_tokens,
    temperature = temperature
  )
  
  return result_query

# COMMAND ----------

# DBTITLE 1,Getting the Insights
question = "How many distinct sensors we have in the bronze table?"

query = get_data_insights(question)

print(f"Generated query: {query}")
output_df = spark.sql(query)
display(output_df)

# COMMAND ----------

question = "What is the average sensor reading value for bronze table?"

query = get_data_insights(question)

print(f"Generated query: {query}")
output_df = spark.sql(query)
display(output_df)

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
# Get metadata

table_metadata = get_table_info(
  catalog = "poc",
  schema = "sensors",
  table = "bronze",
  include_metadata = True
)

prettified_metadata = prettify_metadata(table_metadata)
print(prettified_metadata)

# COMMAND ----------

question = "How many distinct sensors we have in the bronze table?"

query = get_data_insights(
  question = question,
  data_details_instruction = prettified_metadata
)

print(f"Generated query: {query}")
output_df = spark.sql(query)
display(output_df)

# COMMAND ----------

question = "How many sensors in the bronze table contain the substring 'RND'?"

query = get_data_insights(
  question = question,
  data_details_instruction = prettified_metadata
)

print(f"Generated query: {query}")
output_df = spark.sql(query)
display(output_df)

# COMMAND ----------

question = "What is the average sensor reading value for sensors which contain the substring 'RND'?"

query = get_data_insights(
  question = question,
  data_details_instruction = prettified_metadata
)

print(f"Generated query: {query}")
output_df = spark.sql(query)
display(output_df)

# COMMAND ----------

question = "What is the average sensor reading for each different sensor tag name? Group the results by TagName and calculate the average."

query = get_data_insights(
  question = question,
  data_details_instruction = prettified_metadata
)

print(f"Generated query: {query}")
output_df = spark.sql(query)
display(output_df)

# COMMAND ----------

error = """TABLE_OR_VIEW_NOT_FOUND] The table or view `Bronzee` cannot be found. Verify the spelling and correctness of the schema and catalog.
If you did not qualify the name with a schema, verify the current_schema() output, or qualify the name with the correct schema and catalog.
To tolerate the error on drop use DROP VIEW IF EXISTS or DROP TABLE IF EXISTS."""

question = f"""{question}\nDo note that the query below is wrong: '{query}'\nIt returns the following error: '{error}'. Please fix it and generate a new query."""

query = get_data_insights(
  question = question,
  data_details_instruction = prettified_metadata
)

print(f"Generated query: {query}")
output_df = spark.sql(query)
display(output_df)

# COMMAND ----------

error = "[PARSE_SYNTAX_ERROR] Syntax error at or near 'The'.(line 1, pos 0)"

question = f"""{question}\nDo note that the query below is wrong: '{query}'\nIt returns the following error: '{error}'. Please fix it and generate a new query. Return only the query without any explanation. """

query = get_data_insights(
  question = question,
  data_details_instruction = prettified_metadata
)

print(f"Generated query: {query}")
output_df = spark.sql(query)
display(output_df)

# COMMAND ----------

question = f"""The following query returns an error: SELECT TAGNAME,AVG(VALUE)AS AvgReadingFromAllEventsForThatSpecificTagname from (select distinct B.TAGNAME,B.EVENTTIME from BRONZEE groupby B.TAGNAME)t1. The error is the following: '[PARSE_SYNTAX_ERROR] Syntax error at or near 'B': missing ')'.(line 1, pos 139)'. Please generate a new query which fixes this error and return it. Don't include any comments, just return the query."""

query = get_data_insights(question = question)
query

# COMMAND ----------

question = f"""The following query returns an error: {query}. The error is the following: '[MISSING_AGGREGATION] The non-aggregating expression 'Eventtime' is based on columns which are not participating in the GROUP BY clause.
Add the columns or the expression to the GROUP BY, aggregate the expression, or use 'any_value(Eventtime)' if you do not care which of the values within a group is returned.'Please generate a new query which fixes this error and return it. Don't include any introduction or comments, just return the query."""

query = get_data_insights(question = question)
query

# COMMAND ----------


