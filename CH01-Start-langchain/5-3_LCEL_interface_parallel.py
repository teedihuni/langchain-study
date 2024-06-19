from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()
logging.langsmith("CH01-Basic")

model = ChatOpenAI()

# 7. Parallel 병렬

chain1 = (
    PromptTemplate.from_template("{country}의 수도는 어디야?")
    | model
    | StrOutputParser()
)

chain2 = (
    PromptTemplate.from_template("{country}의 면적은 얼마야?")
    |model
    |StrOutputParser()
)

combined = RunnableParallel(capital = chain1, area = chain2)

print(combined.batch([{'country':"대한민국"},{'country':"미국"}]))

