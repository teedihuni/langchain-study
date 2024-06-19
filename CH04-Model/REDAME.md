# CH04 모델

### OPENAI GPT 사용
```
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name = 'gpt-3.5-turbo)

```

### Anthropic Claude3 Sonnet 

```
from langchain_anthropic import ChatAnthropic

# Anthropic 의 Claude 모델 을 생성합니다.
llm = ChatAnthropic(model="claude-3-sonnet-20240229")
```

### llama3-8b 활용
```
from langchain_community.chat_models import ChatOllama

# LangChain이 지원하는 Ollama(로컬) 모델을 사용합니다.
llm = ChatOllama(model="llama3:8b")
```
