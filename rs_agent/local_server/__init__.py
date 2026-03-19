"""
RS-Agent Local LLM Server

폐쇄망 환경에서 Ollama/vLLM 없이 동작하는 OpenAI 호환 LLM 서버.

백엔드 종류:
  mock          - 모델 없이 tool calling 파이프라인 전체 검증 (테스트용)
  transformers  - 로컬 경로의 HuggingFace 모델 사용 (실제 배포용)

사용법:
  python scripts/start_local_server.py --backend mock
  python scripts/start_local_server.py --backend transformers --model-path /models/Llama-3.1-8B-Instruct
"""

from rs_agent.local_server.server import create_app

__all__ = ["create_app"]
