from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH06-Chain')

from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate

template = """
당신은 10년차 엑셀 전문가 입니다. 아래 대화내용을 보고 질문에 대한 적절한 답변을 해주세요

#대화내용
{context}
----
🌱고객: {question}
🤖전문가:"""

prompt = PromptTemplate.from_template(template)

prompt.partial(chat_history="엑셀에서 데이터를 필터링하는 방법에 대해 알려주세요.")

class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}", end="", flush=True)

# LLM 모델 세팅
stream_llm = ChatOpenAI(
    model="gpt-4-turbo-preview", streaming=True, callbacks=[StreamingHandler()]
)

memory = ConversationBufferMemory(
    memory_key="context",  # 대화의 맥락을 저장하는 키
    human_prefix="🌱고객",  # 사람이 말하는 부분에 붙는 접두사
    ai_prefix="🤖전문가",  # AI가 말하는 부분에 붙는 접두사
)

# 대화 세팅
conversation = ConversationChain(
    llm=stream_llm,
    prompt=prompt,
    memory=memory,
    input_key="question",
    verbose=False,
)

print(f'# 초기 메모리 값 : \n{conversation.memory.load_memory_variables({})} \n')

answer = conversation.predict(
    question="엑셀에서 VLOOKUP 함수는 무엇인가요? 간단하게 설명해주세요"
)
print(answer)

print(f'\n # 1번째 대화 이후 메모리 값 : \n{conversation.memory.load_memory_variables({})["context"]} \n')

answer = conversation.predict(
    question="쉬운 예제를 보여줄 수 있나요?"
)
print(answer)

print(f'\n # 2번째 대화 이후 메모리 값 : \n{conversation.memory.load_memory_variables({})["context"]} \n')