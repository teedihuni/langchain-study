import json
import ast

# 텍스트 파일 읽기
with open('prompts/examples.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# JSON 형식으로 변환
data = ast.literal_eval(content)
print(data)

# 결과 출력 (확인용)
# for item in data:
#     print(f"Question: {item['question']}")
#     print(f"Answer: {item['answer']}")
#     print()
