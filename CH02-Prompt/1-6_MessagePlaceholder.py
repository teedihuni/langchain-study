from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
    )

load_dotenv()
logging.langsmith("CH02-Prompt")
llm =ChatOpenAI()

## MessagesPlaceholder 렌더링할 메세지를 제어할 수 있는 기능
# 어떤 역할을 사용해야 할지 확실하지 않거나, 서식 지정중에 메세지 목록을 삽입하려는 경우 유용

human_prompt = 'Summarize our conversation so far in {word_count} words'
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name='conversation'), human_message_template
    ]
)

chat_prompt_print = chat_prompt.format(
    word_count = 5,
    conversation = [
        ("human", "안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다."),
        ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
    ]
)

print(chat_prompt_print)

chain = chat_prompt | llm
result = chain.invoke(
    {
        "word_count": 5,
        "conversation": [
            (
                "human",
                "안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다.",
            ),
            ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
        ],
    }
)

print(result)