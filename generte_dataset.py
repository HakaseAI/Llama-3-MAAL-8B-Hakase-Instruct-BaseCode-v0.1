import os
import json

# 텍스트 파일들이 위치한 디렉토리
directory = './dataset'

# JSON 데이터 저장을 위한 리스트
json_data = []

# 디렉토리 내의 모든 텍스트 파일 읽기
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                # nickname과 chat 분리
                parts = line.strip().split('|')
                if len(parts) == 2:
                    nickname_part, chat_part = parts
                    nickname = nickname_part.split(':')[1]
                    chat = chat_part.split(':')[1]
                    
                    # JSON 데이터로 추가
                    json_data.append({
                        "nickname": nickname,
                        "chat": chat
                    })

# JSON 파일로 저장
with open('dataset/dataset.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 생성되었습니다.")
