from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("CH02-Prompt")


llm =ChatOpenAI()

# 치환될 변수를 {변수}로 묶어서 템플릿을 정의
template = "{country}의 수도는 어디인가요?"

prompt = PromptTemplate.from_template(template)
print(prompt)

# 변수에 값을 직접 입력도 가능
# prompt = prompt.format(country="대한민국")
# print(prompt)

chain = prompt | llm
print(chain.invoke("대한민국").content)
