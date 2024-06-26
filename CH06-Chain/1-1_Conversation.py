from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH06-Chain')

from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

llm =ChatOpenAI()

conversation = ConversationChain( # 대화를 관리
    llm = llm,
    verbose = False, # 상세한 로깅을 위한 설정
    memory= ConversationBufferMemory(memory_key = 'history'), # 대화 내용을 저장할 메모리 버퍼 지정
)

print(conversation.invoke({'input':'양자역학에 대해 설명해줘.'}))
# 대화 추가
conversation.memory.save_context(inputs={"human": "hi"}, outputs={"ai": "안녕"})
print(conversation.memory.load_memory_variables({})["history"])

print(conversation.invoke({"input": "불렛포인트 형식으로 작성해줘. emoji 추가해줘."}))
