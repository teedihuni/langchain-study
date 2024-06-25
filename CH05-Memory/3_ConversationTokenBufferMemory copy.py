from dotenv import load_dotenv
load_dotenv()
from langchain_teddynote import logging
logging.langsmith('CH05-Memory')

from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI


## ConversationTokenBufferMemory : 최근 대화의 히스토리를 버퍼메모리에 보관하고 대화의 개수가 아닌 토큰의 길이를 기준으로 대화내용을 날려버림
# 최대 토큰을 넘어서면 가장 오래된 대화부터 지움

llm = ChatOpenAI()

memory = ConversationTokenBufferMemory(
    llm = llm, max_token_limit= 150, return_messages=True) 

memory.save_context(
    inputs={
        "human": "안녕하세요, 저는 최근에 여러분 회사의 공작 기계를 구매했습니다. 설치 방법을 알려주실 수 있나요?"
    },
    outputs={
        "ai": "안녕하세요! 구매해 주셔서 감사합니다. 해당 기계 모델 번호를 알려주시겠어요?"
    },
)
memory.save_context(
    inputs={"human": "네, 모델 번호는 XG-200입니다."},
    outputs={
        "ai": "감사합니다. XG-200 모델의 설치 안내를 도와드리겠습니다. 먼저, 설치할 장소의 전원 공급 상태를 확인해주세요. 기계는 220V 전원이 필요합니다."
    },
)
memory.save_context(
    inputs={"human": "전원은 확인했습니다. 다음 단계는 무엇인가요?"},
    outputs={
        "ai": "좋습니다. 다음으로, 기계를 평평하고 안정된 바닥에 배치해 주세요. 이후, 제공된 사용자 매뉴얼에 따라 케이블 연결을 진행해 주시기 바랍니다."
    },
)
memory.save_context(
    inputs={"human": "연결은 어떻게 하나요?"},
    outputs={
        "ai": "매뉴얼의 5페이지를 참조해 주세요. 케이블 연결에 관한 상세한 지침이 있습니다. 이 과정에서 어려움이 있으시면 추가적으로 도와드리겠습니다."
    },
)
memory.save_context(
    inputs={"human": "설치가 완료되면 어떻게 해야 하나요?"},
    outputs={
        "ai": "설치가 완료되면, 전원을 켜고 초기 구동 테스트를 진행해 주시기 바랍니다. 테스트 절차는 매뉴얼의 10페이지에 설명되어 있습니다. 만약 기계에 이상이 있거나 추가적인 지원이 필요하시면 언제든지 연락 주시기 바랍니다."
    },
)
memory.save_context(
    inputs={"human": "감사합니다, 도움이 많이 되었어요!"},
    outputs={
        "ai": "언제든지 도와드릴 준비가 되어 있습니다. 추가적인 질문이나 지원이 필요하시면 언제든지 문의해 주세요. 좋은 하루 되세요!"
    },
)

# 기억하고 있는 대화 내용을 확인
# input output 사이에서 짤려 저장될 수 있음
print(f'##최대토큰 길이 150 인 경우 : \n {memory.load_memory_variables({})["history"]}')