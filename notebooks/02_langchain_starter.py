# Databricks notebook source
from models.instruction import InstructionTextGenerationPipeline

from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

# COMMAND ----------

from typing import ClassVar

class InstructionLLM(LLM):

  temperature: ClassVar[float] = 0.3
  top_p: ClassVar[float] = 0.95
  top_k: ClassVar[int] = 0
  max_new_tokens: ClassVar[int] = 500
  _pipeline = InstructionTextGenerationPipeline(model_name="mosaicml/mpt-7b-instruct")
        
  @property
  def _llm_type(self) -> str:
      return "custom"
  
  @classmethod
  def generate(
    self,
    prompt: str,
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    callbacks = None
  ) -> str:
    

    response = self._pipeline.process_stream(
      instruction = prompt,
      temperature = self.temperature,
      top_p = self.top_p,
      top_k = self.top_k,
      max_new_tokens = self.max_new_tokens
    )

    return response
  
  def __call__(
    self,
    prompt
  ):
    return self.generate(prompt = prompt)
  
  def _call(
    self,
    **kwargs
  ):
    pass
  
  @property
  def _identifying_params(self) -> Mapping[str, Any]:
    """Get the identifying parameters."""
    return {
      "temperature": self.temperature,
      "top_p": self.top_p,
      "top_k": self.top_k,
      "max_new_tokens": self.max_new_tokens
    }

# COMMAND ----------

llm = InstructionLLM()

# COMMAND ----------

llm("What is the capital of Lithuania?")

# COMMAND ----------

prompt = """Describe and explain the piece of C++ code below:

#include<iostream>
#include<string.h>
using namespace std;
int main()
{
    char pass[20], ePass[20];
    int numOne, numTwo, sum;
    cout<<"Create a Password: ";
    cin>>pass;
    cout<<"\nEnter Two Numbers to Add: ";
    cin>>numOne>>numTwo;
    cout<<"\nEnter the Password to See the Result: ";
    cin>>ePass;
    if(!strcmp(pass, ePass))
    {
        sum = numOne + numTwo;
        cout<<endl<<numOne<<" + "<<numTwo<<" = "<<sum;
    }
    else
        cout<<"\nSorry! You've entered a Wrong Password!";
    cout<<endl;
    return 0;
}

"""

llm(prompt)

# COMMAND ----------

prompt = """You are an expert in Assembly programming language. Please describe and explain into detail the following piece of Assembly code:

.686
.model flat, stdcall
.stack 4096

extrn	ExitProcess@4: proc				;1 param  1x4
extrn	MessageBoxA@16: proc				;4 params 4x4

.data
	msg_txt	    db	"Hello World", 0
	msg_caption db	"Hello World App", 0

.code
	main:
		push	0				;UINT uType
		lea	eax, msg_caption		;LPCSTR	lpCaption
		push	eax
		lea	eax, msg_txt			;LPCSTR	lpText
		push	eax
		push	0				;HWND hWnd
		call	MessageBoxA@16			;https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-messageboxa
		
		push	0				;UINT uExitCode
		call	ExitProcess@4			;https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-exitprocess
	end main

"""

llm(prompt)

# COMMAND ----------


