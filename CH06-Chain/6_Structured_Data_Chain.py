from dotenv import load_dotenv
load_dotenv()
from langchain_teddynote import logging
logging.langsmith('CH06-Chain')


import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
# from langchain_experimental.tools import PythonAstREPLTool 
# 해당 tool 은 deprecated 됨


