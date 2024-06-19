from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt

import os 
print(os.getcwd())

load_dotenv()
logging.langsmith("CH02-Prompt")
llm =ChatOpenAI()

prompt = load_prompt('prompts/fruit_color.yaml')
