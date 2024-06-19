from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH01-BASIC')

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel

## Parallel 기초
runnable = RunnableParallel(
    # 입력된 데이터를 그대로 통과시키는 역할
    passed = RunnablePassthrough(),
    # 'extra' 키워드 인자로 RunnablePassthrough.assing 을 사용하여 'mult'람다 함수를 할당
    extra = RunnablePassthrough.assign(mult = lambda x : x['num']*3),# 'num'의 키값을 3배 증가
    # 'modified' 키워드 인자로 람다 함수를 전달 
    modified = lambda x : x['num'] +1, # +1
)

print(runnable.invoke({'num':1}))

## Parallel을 chain에 적용
chain1 = (
    {'country':RunnablePassthrough()}
    |PromptTemplate.from_template("{country}의 수도는?")
    |ChatOpenAI()
)

chain2 = (
    {'country':RunnablePassthrough()}
    |PromptTemplate.from_template("{country}의 면적은?")
    |ChatOpenAI()
)
combined_chain = RunnableParallel(capital=chain1, area=chain2)
print(combined_chain.invoke('대한민국'))