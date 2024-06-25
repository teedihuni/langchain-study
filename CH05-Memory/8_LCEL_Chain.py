from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH05-Memory')

from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

model = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful chatbot'),
        MessagesPlaceholder(variable_name='chat_history'),
        ("human","{input}"),
    ]
)

memory = ConversationBufferMemory(
    return_messages=True, memory_key = 'chat_history'
)

print(f'# 초기 메모리 값 : \n{memory.load_memory_variables({})} \n')

runnable = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables)
    | itemgetter("chat_history")  # memory_key 와 동일하게 입력합니다.
)

runnable.invoke({'input':'hi!'})

chain = runnable | prompt | model

response = chain.invoke({'input': '만나서 반갑습니다. 제 이름은 테디입니다.'})
print(f'# 첫번째 대화 \n {response} \n')

# 입력된 데이터와 응답 내용을 메모리에 저장
memory.save_context(
    {"inputs": "만나서 반갑습니다. 제 이름은 테디입니다."}, {"output": response.content}
)

# 저장된 대화기록을 출력합니다.
print(f"# 저장된 대화 기록 \n {memory.load_memory_variables({})} \n")

# 이름을 기억하고 있는지 질문
response = chain.invoke({"input": "제 이름이 무엇이었는지 기억하세요?"})
# 답변을 출력합니다.
print(f'# 이름을 기억하고 있는지 확인 \n {response.content} \n')