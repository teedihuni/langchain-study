from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio

load_dotenv()
logging.langsmith("CH01-Basic")

model = ChatOpenAI()

# 모델 정보 출력
print(f"Default model: {model.model_name if hasattr(model, 'model_name') else 'Unknown'}")
settings = vars(model)
for key, value in settings.items():
    print(f"{key}: {value}")

prompt = PromptTemplate.from_template("{topic}에 대하여 3문장으로 설명해줘.")

chain = prompt | model | StrOutputParser()

# # 1. 실시간 출력
def streaming(chain):
    for token in chain.stream({'topic':"멀티모달"}):
        # 스트림에서 받은 데이터의 내용을 출력
        # 줄바꿈 없이 이어서 출력, 버프를 즉시 비움
        print(token, end="", flush=True)

# 2. invoke 호출
def invoke(chain):
    # 주제를 인자로 받아 처리
    print(chain.invoke({'topic':"ChatGPT"}))

# 3. batch 
# 여러 개의 딕셔너리를 포함하는 리스트를 인자로 받아, 각 딕셔너리의 topic 키값을 사용하여 일괄 처리
#print(chain.batch([{"topic": "ChatGPT"}, {"topic": "Instagram"}]))
def batch(chain):
    print(chain.batch(
        [
            {"topic": "ChatGPT"},
            {"topic": "Instagram"},
            {"topic": "멀티모달"},
            {"topic": "프로그래밍"},
            {"topic": "머신러닝"},
        ],
        config={"max_concurrency": 3}, # 동시에 처리할 수 있는 최대 작업수를 설정
    ))
