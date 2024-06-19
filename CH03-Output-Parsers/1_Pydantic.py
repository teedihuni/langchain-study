from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("CH03-Output-Parsers")

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

llm = ChatOpenAI(temperature=0,
                 model = 'gpt-4-turbo') # 3.5로 하면 원하는 답변이 안나옴...

## [PydanticOutputParser]
# 언어 모델의 출력을 구조화된 정보로 변환하여 사용자가 필요로 하는 정보를 명확하고 체계적인 형태로 제공가능
# * get_format_instructions(): 언어 모델이 출력해야 할 정보의 형식을 정의하는 지침을 제공
# * parse() : 언어 모델의 출력(문자열로 가정)을 받아들여 이를 특정 구조로 분석하고 변환

email_conversation = """From: 김철수 (chulsoo.kim@bikecorporation.me)
To: 이은채 (eunchae@teddyinternational.me)
Subject: "ZENESIS" 자전거 유통 협력 및 미팅 일정 제안

안녕하세요, 이은채 대리님,

저는 바이크코퍼레이션의 김철수 상무입니다. 최근 보도자료를 통해 귀사의 신규 자전거 "ZENESIS"에 대해 알게 되었습니다. 바이크코퍼레이션은 자전거 제조 및 유통 분야에서 혁신과 품질을 선도하는 기업으로, 이 분야에서의 장기적인 경험과 전문성을 가지고 있습니다.

ZENESIS 모델에 대한 상세한 브로슈어를 요청드립니다. 특히 기술 사양, 배터리 성능, 그리고 디자인 측면에 대한 정보가 필요합니다. 이를 통해 저희가 제안할 유통 전략과 마케팅 계획을 보다 구체화할 수 있을 것입니다.

또한, 협력 가능성을 더 깊이 논의하기 위해 다음 주 화요일(1월 15일) 오전 10시에 미팅을 제안합니다. 귀사 사무실에서 만나 이야기를 나눌 수 있을까요?

감사합니다.

김철수
상무이사
바이크코퍼레이션
"""

class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")

# PydanticOutputParser 생성
parser = PydanticOutputParser(pydantic_object=EmailSummary)

prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. Please answer the following questions in KOREAN.

QUESTION:
{question}

EMAIL CONVERSATION:
{email_conversation}

FORMAT:
{format}
"""
)

# format 에 PydanticOutputParser의 format을 추가
prompt = prompt.partial(format=parser.get_format_instructions())

chain = prompt | llm

# chain 을 실행하고 결과를 출력합니다.
response = chain.invoke(
    {
        "email_conversation": email_conversation,
        "question": "이메일 내용중 주요 내용을 추출해 주세요.",
    }
)

# 결과는 JSON 형태로 출력됩니다.
print(response.content)

#print(parser.parse(response.content))

## 출력 파서를 chain에 추가할수도 있음
chain = prompt | llm | parser
# chain 을 실행하고 결과를 출력합니다.
response = chain.invoke(
    {
        "email_conversation": email_conversation,
        "question": "이메일 내용중 주요 내용을 추출해 주세요.",
    }
)

# 결과는 EmailSummary 객체 형태로 출력됩니다.
print(response)