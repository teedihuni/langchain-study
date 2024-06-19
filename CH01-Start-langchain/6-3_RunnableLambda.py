from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH01-BASIC')

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableLambda


def caculate(x):
    print(x)
    ext = x['extra']
    return int(ext['num']*int(ext['mult']))

runnable = RunnableParallel(
    # 입력된 데이터를 그대로 통과시키는 역할
    passed = RunnablePassthrough(),
    # 'extra' 키워드 인자로 RunnablePassthrough.assing 을 사용하여 'mult'람다 함수를 할당
    extra = RunnablePassthrough.assign(mult = lambda x : x['num']*3),# 'num'의 키값을 3배 증가
    # 'modified' 키워드 인자로 람다 함수를 전달 
    modified = lambda x : x['num'] +1, # +1
)

chain = runnable | RunnableLambda(caculate)

# RunnableLambda : 사용자 정의 함수를 맵핑 할 수 있다. 
print(chain.invoke({'num':3}))

## chain 진행 순서 (흐름을 머리속이 아닌 밖으로 이해하기 위해서 한 step씩 적어봄)
# runnable
# 1. runnable 에 num : 3 넣어준다
# 2. passed 는 그냥 값을 그대로 받고
# 3. extra는 lambda로 매핑된 값 num*3 을 전달
# 4. modified는 num+1 값을 전달
    # -- 3번은 새로운 mult를 기존 num과 함께 인자를 추가
    # -- 4번은 변수에 바로 값을 할당
    # -- {'passed': {'num': 3}, 'extra': {'num': 3, 'mult': 9}, 'modified': 4}
# RunnableLamda(calculate)
# 1. 앞에서 정의된 runnable의 값을 가져옴
# 2. 'extra' key에 해당하는 value을 x에 할당
    # -- x = {'num': 3, 'mult': 9}
# 3. 정의한 수식을 바탕으로 3*9 = 27 을 return