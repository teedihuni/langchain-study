from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from datetime import datetime

load_dotenv()
logging.langsmith("CH02-Prompt")
llm =ChatOpenAI()


template = "{country1}과 {country2}의 수도는 각각 어디인가요?"

prompt = PromptTemplate(
    template=template,
    input_variables = ['country1'],# 템플릿 문자열에 있는 변수와 비교하여 불일치 하는 경우 예외를 발생
    partial_variables={
        "country2":'미국'
    }
    )
print('-----기초-----')
print(prompt)
print(prompt.format(country1 = '대한민국'))

prompt_partial = prompt.partial(country = '캐나다')
print(prompt_partial.format(country1 = '대한민국'))

chain = prompt_partial | llm
print(chain.invoke('대한민국').content)
print(chain.invoke({"country1":"대한민국","country2":"호주"}).content)


## 활용
# 프롬프트에 항상 날짜를 넣고 싶다면?
def get_today():
    return datetime.now().strftime("%B %d")

prompt = PromptTemplate(
    template= "오늘의 날짜는 {today} 입니다. 오늘이 생일인 유명인 {n}명을 나열해주세요. 생년월일도 표기해주세요",
    input_variables=['n'],
    partial_variables={
        'today':get_today # dictionary 형태로 partial_variables로 전달
    }
)
print('-----응용-----')
print(prompt.format(n=3))

chain = prompt | llm
print(chain.invoke(3).content)
print(chain.invoke({'today':'Jan 02','n':3}).content)
