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


# 4. async stream : chain.asteram
# 비동기 스트림을 사용하여 'YouTube' 토픽의 메시지를 처리합니다.
async def async_stream(chain):
        
    async for token in chain.astream({"topic": "YouTube"}):
        # 메시지 내용을 출력합니다. 줄바꿈 없이 바로 출력하고 버퍼를 비웁니다.
        print(token, end="", flush=True)
# 5. async invoker
async def asycn_invoker(chain) :
    my_process = await chain.ainvoke({'topic':"NVDA"})
    print(my_process)

# 6. async batch :비동기 배치
# 비동기적으로 일련의 작업을 일괄 처리

async def async_batch(chain):
    my_abatch_process = await chain.abatch(
        [{"topic": "YouTube"}, {"topic": "Instagram"}, {"topic": "Facebook"}]
    )
    print(my_abatch_process)

# 실행하려는 함수 입력 설정 후 실행
asyncio.run(async_stream(chain))


