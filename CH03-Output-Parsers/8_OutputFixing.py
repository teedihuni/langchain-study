from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

load_dotenv()
logging.langsmith("CH03-Output-Parsers")

class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(
        description="list of names of films they starred in")
    
actor_query = "Generate the filmography for a random actor."

parser = PydanticOutputParser(pydantic_object=Actor)

# 잘못된 형식을 일부러 입력
misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"

# 잘못된 형식으로 입력된 데이터를 파싱하려고 시도
#parser.parse(misformatted)

# 오류 출력

from langchain.output_parsers import OutputFixingParser

## [OutputFixingParser]
# 출력 파싱 과정에서 발생할 수 있는 오류를 자동으로 수정

new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())

actor = new_parser.parse(misformatted)
print(actor)