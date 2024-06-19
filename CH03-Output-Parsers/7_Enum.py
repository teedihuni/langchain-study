from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers.enum import EnumOutputParser
from enum import Enum

load_dotenv()
logging.langsmith("CH03-Output-Parsers")


class Colors(Enum):
    RED = "빨간색"
    GREEN = "초록색"
    BLUE = "파란색"

parser = EnumOutputParser(enum=Colors)

prompt = PromptTemplate.from_template(
    """다음의 물체는 어떤 색깔인가요?

Object : {object}

Istructions : {instructions}"""
).partial(instructions=parser.get_format_instructions())

chain = prompt | ChatOpenAI() | parser

response = chain.invoke({"object": "하늘"})  # "하늘" 에 대한 체인 호출 실행
print(response)