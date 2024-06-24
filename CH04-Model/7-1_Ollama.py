## Ollama 를 사용하면 LLaMA2와 같은 오픈 소스 LLM을 로컬에서 실행할 수 있다.
## 지원 모델 목록 https://ollama.com/library
## ollam pull <model_name> 할 때 특정 파라미터의 모델을 명시하지 않으면 가장 작은 모델을 가져온다.

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model = 'qwen2')
# print(f"Default model: {llm.model_name if hasattr(llm, 'model_name') else 'Unknown'}")
# settings = vars(llm)
# for key, value in settings.items():
#     print(f"{key}: {value}")


prompt = ChatPromptTemplate.from_template("{topic}에 대해서 간략히 설명해줘.")

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic" : "deep learning"}))

topic = {'topic' : "Covid 19"}

for chunks in chain.stream(topic):
    print(chunks, end ="", flush = True)