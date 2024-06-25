from dotenv import load_dotenv
load_dotenv()
from langchain_teddynote import logging
logging.langsmith('CH05-Memory')

from langchain_openai import ChatOpenAI
from langchain_community.memory.kg import ConversationKGMemory # 이렇게 수정하라고 함

## ConversationKGMemory : 지식 그래프를 활용하여 정보를 저장
# 모델이 서로 다른 개체 간의 관계를 이해하는데 도움을 주고, 복잡한 연결망과 역사적 맥락을 기반으로 대응하는 능력을 향상

# 기초 예시
llm = ChatOpenAI(temperature=0)
memory = ConversationKGMemory(llm=llm, return_messages=True)

memory.save_context(
    {"input": "이쪽은 Pangyo 에 거주중인 김셜리씨 입니다."},
    {"output": "김셜리씨는 누구시죠?"},
)
memory.save_context(
    {"input": "김셜리씨는 우리 회사의 신입 디자이너입니다."},
    {"output": "만나서 반갑습니다."},
)

print(memory.load_memory_variables({"input": "김셜리씨는 누구입니까?"}))

# 체인 지정
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.base import ConversationChain

llm = ChatOpenAI(temperature=0)

template = """The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. 
The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=template)

conversation_with_kg = ConversationChain(
    llm=llm, prompt=prompt, memory=ConversationKGMemory(llm=llm)
)

print(
    conversation_with_kg.predict(
    input="My name is Teddy. Shirley is a coworker of mine, and she's a new designer at our company.")
    )

# Shirley 에 대한 질문
print(
    conversation_with_kg.memory.load_memory_variables({"input": "who is Shirley?"})
    )
