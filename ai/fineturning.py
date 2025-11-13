import openai
import datasets
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)
dataset = openai.files.create(
    file=open('/Users/choijiwon/Desktop/dcon/prompt2.jsonl', "rb"),
    purpose="fine-tune"
)
fine_tune = openai.fine_tuning.jobs.create(
    training_file=dataset.id,
    model="gpt-4o-mini-2024-07-18"
)
#학습 상태 확인하기
status = openai.fine_tuning.jobs.list(limit=20)
print(status.data[0].status)