from dotenv import load_dotenv
load_dotenv()
from langchain_teddynote import logging 
logging.langsmith('CH05-Memory')

from langchain_community.chat_message_histories import SQLChatMessageHistory

## sql 객체 생성
chat_message_history = SQLChatMessageHistory(
    session_id='sql_chat_history', connection='sqlite:///sqlite.db'
)

# 사용자 메세지 추가
chat_message_history.add_user_message(
    "Hi, My name is Teddy. I am a AI programmer. Nice to meet you"
)

# AI 메시지 추가
chat_message_history.add_ai_message("Hi Teddy! Nice to meet you too!")
print(f'초기 히스토리 {chat_message_history.messages} \n')

## chain 적용
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        # 시스템 메시지를 설정하여 어시스턴트의 역할을 정의
        ("system", "You are a helpful assistant."),
        # 이전 대화 내용을 포함하기 위한 플레이스홀더 추가
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),  
    ]
)

chain = (
    prompt | ChatOpenAI()
) 

# chain과 메시지 기록을 연결
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection="sqlite:///sqlite.db"
    ),  # session_id를 기반으로 SQLChatMessageHistory 객체를 생성하는 람다 함수
    input_messages_key="question",  # 입력 메시지의 키를 "question"으로 설정
    history_messages_key="history",  # 대화 기록 메시지의 키를 "history"로 설정
)

config = {"configurable": {"session_id": "sql_chat_history"}}
print(chain_with_history.invoke({"question": "Whats my name?"}, config=config))