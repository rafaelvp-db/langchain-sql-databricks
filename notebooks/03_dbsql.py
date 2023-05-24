# Databricks notebook source
from llms.instruction import InstructionLLM

# COMMAND ----------

llm = InstructionLLM()

# COMMAND ----------

# DBTITLE 1,Test 1.1: Simple Querying
# Here we define a standard prompt, where we give some of the details on where to find the data for getting results
# We also explicitly define which catalog we're interested in, though we might skip that step later on if we're able
# to fetch UC data in realtime

def get_data_insights(question, temperature = 0, max_new_tokens = 1000):

  # Set UC catalog
  spark.sql("use catalog poc")
  spark.sql("use schema sensors")

  data_details_instruction = """
    We have two tables, bronze and silver, and they are both part of the sensors schema.

    In the bronze table, we have the following columns:
      TagName (string): specifies the sensor ID
      EventTime (timestamp): timestamp during which the event occurred
      Status (string): Can be either 'Good' or 'Bad'
      Value (float): Contains the sensor reading for that particular point in time
  """

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


