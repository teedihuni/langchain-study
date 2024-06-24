## 출력의 format을 설정해보자

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

import json

llm = ChatOllama(model = 'qwen2',
                 format= 'json',
                 temperature=0)

# 예시 1
messages = [
    HumanMessage( #사용자의 질문을 담은 메세지 객체를 새엇ㅇ
        content = 'Tell me 10 places to travel in Europe. response in JSON format'
    )
]

chat_model_response = llm.invoke(messages)
print(f"### 예시 1번 출력 \n {chat_model_response.content}")

# 예시 2
json_schema = {
    "title": "Person",
    "description": "Identifying information about a person.",
    "type": "object",
    "properties": {
        "name": {"title": "Name", "description": "The person's name", "type": "string"},
        "age": {"title": "Age", "description": "The person's age", "type": "integer"},
        "occupation": {
            "title": "Occupation",
            "description": "The person's Occupation",
            "type": "string",
        },
    },
    "required": ["name", "age"],
}

messages = [
    HumanMessage(
        # JSON 스키마를 사용하여 사람에 대해 설명해달라는 요청 메시지
        content="Please tell me about a person using the following JSON schema:"
    ),
    HumanMessage(content="{dumps}"),  # JSON 스키마를 메시지로 전달
    HumanMessage(
        # 스키마를 고려하여 John이라는 35세의 피자를 좋아하는 사람에 대해 설명해달라는 요청 메시지
        content="""Now, considering the schema, please describe following person:
        Her name is Eun-Chae Lee, she is 25 years old, and she is a software engineer.
        """
    ),
]

prompt = ChatPromptTemplate.from_messages(
    messages
)

dumps = json.dumps(json_schema, indent=2) #json 스키마를 문자열로 변환

chain = (
    prompt | llm | StrOutputParser()
)

print(f"### 예시 2번 출력 \n {chain.invoke({'dumps':'dumps'})}")