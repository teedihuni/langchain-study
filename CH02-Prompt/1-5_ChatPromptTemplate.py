from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
logging.langsmith("CH02-Prompt")
llm =ChatOpenAI()

## ChatPromptTemplate는 대화목록을 프롬프트로 주입하고자 할 때 사용한다.
# 메세지는 튜블 형식으로 구성하며, (role, message)로 구성한다
# - role : 'system' 설정 메세지로 전역설정과 관련됨
# - human : 사용자 입력 메세지
# - ai : AI 답변 메세지
chat_template = ChatPromptTemplate.from_messages(
    [
        # role, message
        ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name} 입니다."),
        ("human", "반가워요!"),
        ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
        ("human", "{user_input}"),
    ]
)

# message 생성
messages = chat_template.format_messages(
    name="테디", user_input="당신의 이름은 무엇입니까?"
)
print(messages)
print(llm.invoke(messages).content)

# chain 설정
chain = chat_template | llm
print(chain.invoke({"name":"Huni","user_input":'당신의 이름은 무엇입니까?'}).content)