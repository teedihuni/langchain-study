from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH06-Chain')

from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks.base import BaseCallbackHandler


class StreamingHandler(BaseCallbackHandler): # 해당 함수에서 print 해줌
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}", end="", flush=True)

llm =ChatOpenAI(streaming=True, callbacks=[StreamingHandler()])


conversation = ConversationChain( # 대화를 관리
    llm = llm,
    verbose = False, # 상세한 로깅을 위한 설정
    memory= ConversationBufferMemory(), # 대화 내용을 저장할 메모리 버퍼 지정
)

output = conversation.predict(input='양자역학에 대해 설명해줘.')
print(output)

output = conversation.predict(
    input="이전의 내용을 불렛포인트로 요약해줘. emoji 추가해줘."
)
print(output)
