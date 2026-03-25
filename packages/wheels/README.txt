이 디렉토리에는 인터넷 없이 설치 가능한 코어 패키지 wheel 파일이 포함되어 있습니다.

포함된 패키지:
  numpy, scikit-learn, faiss-cpu, Pillow, PyYAML, rank-bm25, rich,
  python-dotenv, requests, openai, anthropic, fastapi, uvicorn, tiktoken

미포함 패키지 (용량 초과로 git 제외, rs-agent-env.tar.gz에 포함됨):
  torch (~500MB), transformers, gradio, sentence-transformers, bitsandbytes

사용법:
  pip install --no-index --find-links packages/wheels/ <패키지명>
