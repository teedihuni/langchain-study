from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH01-BASIC')

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

'''
RunnablePassthrough 는 입력을 변경하지 않거나 추가 키를 더하여 전달할 수 있습니다.
RunnablePassthrough() 가 단독으로 호출되면, 단순히 입력을 받아 그대로 전달합니다.
RunnablePassthrough.assign(...) 방식으로 호출되면, 입력을 받아 assign 함수에 전달된 추가 인수를 추가합니다.

'''
# 초기 모델 세팅
prompt = PromptTemplate.from_template('{num}의 10배는?')
llm = ChatOpenAI(temperature=0)

# chain 생성
chain = prompt | llm 

# print(chain.invoke({'num':5}))
# print(chain.invoke(5)) # 값만 넣어줘도 invoke 가능

runnable_chain = {'num': RunnablePassthrough()}| prompt | ChatOpenAI()
#print(runnable_chain.invoke(10))
print(RunnablePassthrough.assign(new_num=lambda x : x["num"]*3).invoke({'num':1}))