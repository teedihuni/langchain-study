from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_teddynote import logging
import time
logging.langsmith('CH04-Model')

llm = ChatOpenAI()

prompt = PromptTemplate.from_template('{country} 에 대해서 200자 내외로 요약해줘')
print(prompt)

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time:.4f} seconds")
        return result
    return wrapper

@time_it
def invoke_chain():
    return chain.invoke({"country": "한국"})

chain = prompt | llm
response = invoke_chain()
print(response.content)


## 똑같은 요청을 다시 하는 경우 cache를 통해서 API 비용을 아낄수 있다
from langchain_community.cache import SQLiteCache #기존 튜토리얼의 코드에서 이와같이 변경됨
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path='my_llm_cache.db'))
response = invoke_chain()
print(response.content)

