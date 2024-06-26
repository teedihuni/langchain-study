from dotenv import load_dotenv
load_dotenv()
from langchain_teddynote import logging
logging.langsmith('CH07-Document_Loader')


from langchain_community.document_loaders import ArxivLoader
# ArxivLoader를 사용하여 arXiv에서 문서를 로드합니다. 
# query 매개변수는 검색할 논문의 arXiv ID이고
# load_max_docs 매개변수는 로드할 최대 문서 수를 지정합니다.
docs = ArxivLoader(query="1605.08386", load_max_docs=2).load()
len(docs)  # 로드된 문서의 개수를 반환

print(f">> 0번 문서의 메타 속성 >> \n {docs[0].metadata}") # meta속성은 document와 관련된 메타데이터 정보를 return
print(f">> 0번 문서의 페이지 중 일부 추출  \n {docs[0].page_content[:400]}")  # 문서의 모든 페이지 내용 중 처음 400자를 return
print(f">> 0번 문서의 페이지 수  \n {len(docs[0].page_content)}") # 문서의 페이지 수