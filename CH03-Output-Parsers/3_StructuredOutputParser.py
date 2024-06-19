from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate


load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("CH03-Output-Parsers")


## [StructuredOutputParser]
# 여러 필드를 반환하고자 할때 사용

# 사용자의 질문에 대한 답변
response_schemas = [
    ResponseSchema(name="answer", description="사용자의 질문에 대한 답변"), # 이러한 구조로 답변해라~ 라고 설정해주는것임
    ResponseSchema(
        name="source",
        description="사용자의 질문에 답하기 위해 사용된 출처, 웹사이트 이여야 합니다.",
    ),
]
# 응답 스키마를 기반으로 한 구조화된 출력 파서 초기화
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    # 사용자의 질문에 최대한 답변하도록 템플릿을 설정합니다.
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    # 입력 변수로 'question'을 사용합니다.
    input_variables=["question"],
    # 부분 변수로 'format_instructions'을 사용합니다.
    partial_variables={"format_instructions": format_instructions},
)

print(prompt)

model = ChatOpenAI(temperature=0)
chain = prompt | model | output_parser

result1 = chain.invoke({'question':"대한민국의 수도는 어디인가요?"})
result2= chain.stream({"question":"세종대왕의 업적은 무엇인가요?"})

print(result1)
for s in result2:
    print(s)
