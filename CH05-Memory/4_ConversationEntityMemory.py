from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH05-Memory')

from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

## ENTITY Memory를 사용하는 프롬프트 내용을 출력
print(ENTITY_MEMORY_CONVERSATION_TEMPLATE.template)

## ENTITY_MEMORY_CONVERSATION_TEMPLATE
# 엔티티 메모리는 대화에서 특정 엔티티에 대한 주어진 사실을 기억한다
# llm을 통해 정보를 추출하고 시간이 지남에 따라 해당 엔티티에 대한 지식을 축적한다.


llm = ChatOpenAI(temperature=0)
conversation = ConversationChain(
    llm = llm,
    prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm = llm)
)

response = conversation.predict(
    input="테디와 셜리는 한 회사에서 일하는 동료입니다."
    "테디는 개발자이고 셜리는 디자이너입니다. "
    "그들은최근 회사에서 일하는 것을 그만두고 자신들의 회사를 차릴 계획을 세우고 있습니다."
)

print(f'## 결과 출력 : \n {response}')
print(f"## 엔티티 주요 정보 출력 : \n {conversation.memory.entity_store.store}")

