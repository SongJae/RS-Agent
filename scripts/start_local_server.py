#!/usr/bin/env python3
"""
RS-Agent 로컬 LLM 서버 실행 스크립트

폐쇄망 환경에서 Ollama/vLLM 대신 사용하는 OpenAI 호환 로컬 서버입니다.

사용 예:
  # Mock 모드 (모델 없이 파이프라인 검증)
  python scripts/start_local_server.py --backend mock

  # Transformers 모드 (로컬 모델 사용)
  python scripts/start_local_server.py --backend transformers \\
    --model-path /models/Qwen2.5-7B-Instruct

  # CPU 전용 + 8비트 양자화
  python scripts/start_local_server.py --backend transformers \\
    --model-path /models/Llama-3.1-8B-Instruct \\
    --device cpu --load-in-8bit
"""

import argparse
import logging
import os
import sys

# 폐쇄망 환경변수 설정
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="RS-Agent 로컬 LLM 서버")
    parser.add_argument(
        "--backend",
        choices=["mock", "transformers"],
        default="mock",
        help="LLM 백엔드 선택 (default: mock)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="[transformers 백엔드] 로컬 모델 디렉토리 경로",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="[transformers 백엔드] 연산 디바이스 (auto/cpu/cuda, default: auto)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="[transformers 백엔드] 8비트 양자화 사용 (GPU 메모리 절약)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="서버 바인딩 주소 (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="서버 포트 (default: 11434, Ollama와 동일)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("rs_agent.server")

    try:
        import uvicorn
        from rs_agent.local_server.server import create_app
    except ImportError as e:
        logger.error(f"필수 패키지 없음: {e}")
        logger.error("pip install fastapi uvicorn 실행 후 다시 시도하세요.")
        sys.exit(1)

    backend_kwargs = {
        "model_path": args.model_path,
        "device": args.device,
        "load_in_8bit": args.load_in_8bit,
    }

    logger.info(f"백엔드: {args.backend}")
    if args.backend == "transformers":
        if not args.model_path:
            logger.error("--model-path 옵션이 필요합니다.")
            sys.exit(1)
        logger.info(f"모델 경로: {args.model_path}")

    app = create_app(backend_name=args.backend, **backend_kwargs)

    logger.info(f"서버 시작: http://{args.host}:{args.port}")
    logger.info("RS-Agent config.yaml의 llm.base_url을 아래로 설정하세요:")
    logger.info(f"  base_url: \"http://localhost:{args.port}/v1\"")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
