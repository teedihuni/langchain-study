from dotenv import load_dotenv
load_dotenv()
from langchain_teddynote import logging
logging.langsmith('CH05-Memory')

from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain # from 부분이 튜토리얼대로하면 import가 안됨
from langchain.memory import ConversationBufferMemory


## ConversationBufferMemory : 메세지를 저장한 다음 변수에 메세지를 출력할 수 있도록 해준다.

# 예시 1
memory = ConversationBufferMemory() 
memory2 = ConversationBufferMemory(return_messages=True) 
memory.save_context( 
    # save(inputs, outputs)를 통해 대화기록을 저장
    # 대화기록은 history key에 저장된다
    inputs={
        'human':'안녕하세요, 비대면으로 은행계좌를 개설하고 싶습니다. 어떻게 시작해야하나요?'
    },
    outputs = {
        "ai": "안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?"}
)
memory2.save_context( 
    # save(inputs, outputs)를 통해 대화기록을 저장
    # 대화기록은 history key에 저장된다
    inputs={
        'human':'안녕하세요, 비대면으로 은행계좌를 개설하고 싶습니다. 어떻게 시작해야하나요?'
    },
    outputs = {
        "ai": "안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?"}
)

print(f"##1번 예시 -  return message False 인 경우: \n {memory.load_memory_variables({})}")
print(f"##2번 예시 - return message True 인 경우: \n {memory2.load_memory_variables({})}")

# 예시 2 - 2개의 대화를 추가로 저장하는 경우
memory.save_context(
    inputs={"human": "사진을 업로드했습니다. 본인 인증은 어떻게 진행되나요?"},
    outputs={
        "ai": "업로드해 주신 사진을 확인했습니다. 이제 휴대폰을 통한 본인 인증을 진행해 주세요. 문자로 발송된 인증번호를 입력해 주시면 됩니다."
    },
)
memory.save_context(
    inputs={"human": "인증번호를 입력했습니다. 계좌 개설은 이제 어떻게 하나요?"},
    outputs={
        "ai": "본인 인증이 완료되었습니다. 이제 원하시는 계좌 종류를 선택하고 필요한 정보를 입력해 주세요. 예금 종류, 통화 종류 등을 선택할 수 있습니다."
    },
)

print(f"##3번 예시 - 2개 대화 추가하는 경우 : \n {memory.load_memory_variables({})['history']}")

## Chain에 적용
llm = ChatOpenAI(temperature=0)

conversation = ConversationChain(
    llm = llm,
    memory = ConversationBufferMemory()
)

response= conversation.predict(
    input = "안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
)
print(f"##4번 예시 - chain 추론 : \n {response}")