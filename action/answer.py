from typing import List
from transformers import set_seed
import re
import json
import logging
from src.promptTemplate import answer_generation_prompt_template


class Answer(object):
    def __init__(self, llm) -> None:
        super().__init__()
        self.class_name = 'Answer_Generator'
        self.class_desc = 'Using this action to generate answer directly.'
        self.llm = llm
        set_seed(42)
        
    def call(self, filtered_triplets, query):

        prompt = answer_generation_prompt_template.replace('{t}', filtered_triplets).replace('{q}', query)
        outputs = self.llm.generate(prompt, new_tokens_num=30)

        return outputs  