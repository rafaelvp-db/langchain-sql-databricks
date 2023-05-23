# Databricks notebook source
from typing import Any, Dict, Tuple
import warnings
import datetime
import os
from threading import Event, Thread
import torch
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  StoppingCriteria,
  StoppingCriteriaList,
  TextIteratorStreamer
)

import textwrap

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

class InstructionTextGenerationPipeline:
    def __init__(
        self,
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=None,
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )
        if tokenizer.pad_token_id is None:
            warnings.warn(
                "pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id."
            )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device=device, dtype=torch_dtype)

        self.generate_kwargs = {
            "temperature": 0.5,
            "top_p": 0.92,
            "top_k": 0,
            "max_new_tokens": 512,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.1,  # 1.0 means no penalty, > 1.0 means penalty, 1.2 from CTRL paper
        }

    def format_instruction(self, instruction):
        return PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)

    def __call__(
        self, instruction: str, **generate_kwargs: Dict[str, Any]
    ) -> Tuple[str, str, float]:
        s = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)
        input_ids = self.tokenizer(s, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        gkw = {**self.generate_kwargs, **generate_kwargs}
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gkw)
        # Slice the output_ids tensor to get only new tokens
        new_tokens = output_ids[0, len(input_ids[0]) :]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return output_text

# COMMAND ----------

# Define a custom stopping criteria
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def process_stream(instruction, temperature, top_p, top_k, max_new_tokens):
    # Tokenize the input
    input_ids = generate.tokenizer(
        generate.format_instruction(instruction), return_tensors="pt"
    ).input_ids
    input_ids = input_ids.to(generate.model.device)

    # Initialize the streamer and stopping criteria
    streamer = TextIteratorStreamer(
        generate.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    stop = StopOnTokens()

    if temperature < 0.1:
        temperature = 0.0
        do_sample = False
    else:
        do_sample = True

    gkw = {
        **generate.generate_kwargs,
        **{
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([stop]),
        },
    }

    response = ''
    
    def generate_and_signal_complete():
        generate.model.generate(**gkw)

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    for new_text in streamer:
        response += new_text
   
    return response

# COMMAND ----------

import transformers

generate = InstructionTextGenerationPipeline(
    "mosaicml/mpt-7b-instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# COMMAND ----------

from pyspark.errors import AnalysisException

# COMMAND ----------

stop_token_ids = generate.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

DEFAULT_PROMPT = """
The 'bronze' table is part of the 'sensors' schema, and the 'poc' catalog. The SQL DDL code to create the 'poc.sensors.bronze' table is the following:

CREATE TABLE poc.sensors.bronze (
  EventTime timestamp,
  Value float,
  Status string,
  TagName string
)

Possible values for 'Status' are 'Good' and 'Bad' and these are case sensitive.

Assume you are a SQL expert. Write a SQL query to answer the following question about the bronze table: {question}. Write the query without any comments. When including string columns in the WHERE clause, make sure to sanitize the input by converting the column to lower case, for example: 'WHERE lower(Status) = lower('Good')'
"""

def get_query(
  question: str,
  temperature = 0.3,
  top_p = 0.99,
  top_k = 1,
  max_new_tokens = 500
):
  
  prompt = DEFAULT_PROMPT.format(
    question = question
  )

  response = process_stream(
    prompt,
    temperature,
    top_p,
    top_k,
    max_new_tokens
  )

  return response

def get_answer(query):

  df = spark.sql(query)
  return df

def run_chain(question, current_tries = 0, max_tries = 10):

  current_tries_local = current_tries
  if current_tries < max_tries:
    try:
        current_tries_local += 1
        query = get_query(question)
        print(f"Attempt {current_tries}, query: {query}")
        result = get_answer(query)
        numeric_result = result.toPandas().values[0]
        print(numeric_result)
        if numeric_result == 0:
          raise ValueError(f"The query returned zero rows")
        display(result)

    except Exception as exception:
      message = str(exception)
      new_question = DEFAULT_PROMPT + f""" For example, the query '{query}' is incorrect, it generates the following error: {message}. Please re-write this SQL query. Don't include any comments or explanations, just return the SQL query."""
      run_chain(question = new_question, current_tries = current_tries_local)

  else:
    print("Max tries reached")


# COMMAND ----------

question = "What is the average Value for sensor readings?"

run_chain(question)

# COMMAND ----------

spark.sql("use catalog poc")

question = "How many rows are there where Status = 'Good'?"

run_chain(question)

# COMMAND ----------

question = "Calculate how many rows are there where Status = 'Good', group by TagName"

run_chain(question)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT count(*), tagname FROM sensors.bronze WHERE LOWER(STATUS) = LOWER('GOOD') GROUP BY TAGNAME

# COMMAND ----------



