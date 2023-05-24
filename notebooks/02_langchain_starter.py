# Databricks notebook source
from llms.instruction import InstructionLLM

# COMMAND ----------

# DBTITLE 1,Instantiate our LLM object
llm = InstructionLLM()

# COMMAND ----------

# DBTITLE 1,Test 1: Travel Guide
prompt = "Write me a four day trip guide for Vilnius, Lithuania, during summer. Please include some Michelin star restaurants if possible. Write the instructions separately for each day."

trip_guide = llm(prompt, max_new_tokens = 1000)
print(trip_guide)

# COMMAND ----------

# DBTITLE 1,Test 2: Explain C++ Code
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

# DBTITLE 1,Test 3: Explain Assembly Code
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

# DBTITLE 1,Test 4.1: Explain SAS Code
code = """

proc delete data=mydblib.NEBLKTAB; run;

data work.NEBLKDAT;
   input name $ age sex $ bdate mmddyy.;
   cards;
amy 3 f 030185
bill 12 m 121277
charlie 35 m 010253
david 19 m 101469
elinor 42 f 080845
pearl 78 f 051222
vera 96 f 101200
frank 24 m 092663
georgia 1 f 040687
henry 46 m 053042
joann 27 f 020461
buddy 66 m 101432
;
run;

proc sql;
create table mydblib.NEBLKTAB (BULKLOAD=YES) as
   select * from work.NEBLKDAT;
quit;

proc print data=mydblib.NEBLKTAB(BULKUNLOAD=YES);
 format bdate date7.;
title 'proc print of table';
run;
"""

prompt = f"""The following piece of code contains SAS logic to connect to Netezza. Please explain this SAS piece of code into detail:

{code}
"""

# COMMAND ----------

comments = llm(prompt)
comments

# COMMAND ----------

# DBTITLE 1,Test 4.2: Convert SAS Code to PySpark
prompt = f"""The following piece of code contains SAS logic:

{code}

The explanation for the SAS code can be found below:

'{comments}'

Please re-write this piece of SAS code in Python using the PySpark library. and display only the resulting code, without any text or comments.

"""

python_code = llm(prompt, max_new_tokens=10000)
print(python_code)

# COMMAND ----------

print(python_code)

# COMMAND ----------

# DBTITLE 1,Test 4.3: Save PySpark code and Lint it
import subprocess
from ruff import __main__ as r

python_code = "#" + python_code

with open("/tmp/test.py", "w") as output_file:
  output_file.write(python_code)

ruff_bin = r.find_ruff_bin()
result = subprocess.run([ruff_bin, "/tmp/test.py", "--fix-only"], stdout=subprocess.PIPE)
result.stdout

with open("/tmp/test.py") as fixed_code:
  print(fixed_code.read())

# COMMAND ----------

# DBTITLE 1,Test 5: Drawing
prompt = "You are a Machine Learning specialist. Please write a Computer Vision Neural Network in PyTorch Lightning for Image Classification. Include the complete code, without any placeholders."

text_output = llm(prompt, max_new_tokens = 10000)

print(text_output)
