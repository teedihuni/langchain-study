from langchain import hub

prompt = hub.pull('rlm/rag-prompt')
print(prompt)

from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "주어진 내용을 바탕으로 다음 문장을 요약하세요. 답변은 반드시 한글로 작성하세요\n\nCONTEXT: {context}\n\nSUMMARY:"
)
print(prompt)

# hub.push('huni/simple-summary-korean',prompt)

# hub에 대한 부분은 조금 더 서칭이 필요함