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

# Define a custom stopping criteria
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
      super().__init__()
      self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class InstructionTextGenerationPipeline:
    def __init__(
        self,
        model_name = "mosaicml/mpt-7b-instruct",
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

    def process_stream(
      self,
      instruction,
      temperature,
      top_p,
      top_k,
      max_new_tokens
    ):
        # Tokenize the input
        input_ids = self.tokenizer(
            self.format_instruction(instruction), return_tensors="pt"
        ).input_ids
        input_ids = input_ids.to(self.model.device)

        # Initialize the streamer and stopping criteria
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=10.0,
            skip_prompt=True,
            skip_special_tokens=True
        )

        stop = StopOnTokens(self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"]))

        if temperature < 0.1:
            temperature = 0.0
            do_sample = False
        else:
            do_sample = True

        gkw = {
            **self.generate_kwargs,
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
            self.model.generate(**gkw)

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        for new_text in streamer:
            response += new_text
      
        return response