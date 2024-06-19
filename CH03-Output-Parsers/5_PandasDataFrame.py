from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain.output_parsers import (
    PandasDataFrameOutputParser,
)  # Pandas 데이터프레임 출력 파서

import pprint
from typing import Any, Dict

load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("CH03-Output-Parsers")


# format_parser_output : 파서 출력을 사전 형식으로 변환하고 이쁘게 출력
model = ChatOpenAI(temperature=0)

def format_parser_output(parser_output:Dict[str, Any]) -> None:
    for key in parser_output.keys():
        parser_output[key] = parser_output[key].to_dict()
    
    return pprint.PrettyPrinter(width=4, compact=True).pprint(parser_output)

df = pd.read_csv('./data/titanic.csv')
#print(df.head())

parser = PandasDataFrameOutputParser(dataframe=df)

# 열 작업 예시
df_query = "Retrieve the passengers age."
df_query = "Retrieve the first row."
df_query = "Retrieve the average of the age from row 0 to 4"

# 프롬프트 설정
prompt = PromptTemplate(
    template='Answer the user query.\n{format_instructions}\n{query}\n',
    input_variables=["query"],
    partial_variables={
        "format_instructions":parser.get_format_instructions()
    },
)

chain = prompt | model | parser
parser_output = chain.invoke({'query':df_query})

try :
    format_parser_output(parser_output)
except:
    print(parser_output)



