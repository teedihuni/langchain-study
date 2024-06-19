from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH01-BASIC')

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate

def length_function(text):
    return len(text)

def _multiple_length_function(text1, text2):
    return len(text1)*len(text2)

def multiple_length_function(_dict):
    return _multiple_length_function(_dict['text1'], _dict['text2'])

prompt = ChatPromptTemplate.from_template("{a} + {b} 는 무엇인가요?")
model = ChatOpenAI()

#chain1 = prompt | model

chain = (
    {
        'a' : itemgetter('word1') | RunnableLambda(length_function), # word1의 길이
        'b' : {'text1' : itemgetter('word1'), 'text2' : itemgetter('word2')} 
        | RunnableLambda(multiple_length_function), #word1의 길이 * word2의 길이
    }
    | prompt
    | model
    | StrOutputParser()
)

print(chain.invoke({'word1': 'hello','word2':'world'}))