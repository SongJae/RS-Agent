#!/usr/bin/env python3
"""
RS-Agent 로컬 LLM 서버 실행 스크립트

폐쇄망 환경에서 Ollama/vLLM 대신 사용하는 OpenAI 호환 로컬 서버입니다.

[Mock 모드] 모델 없이 파이프라인 검증 (GPU 불필요):
  python scripts/start_local_server.py --backend mock

[A100 0.3장, 24GB] Qwen2.5-7B FP16:
  python scripts/start_local_server.py --backend transformers \\
    --model-path /models/Qwen2.5-7B-Instruct

[A100 0.3장, 12GB] Qwen2.5-7B 4-bit 양자화:
  python scripts/start_local_server.py --backend transformers \\
    --model-path /models/Qwen2.5-7B-Instruct --load-in-4bit

[GPU 메모리 절약, 8-bit]:
  python scripts/start_local_server.py --backend transformers \\
    --model-path /models/Llama-3.1-8B-Instruct --load-in-8bit

[CPU 전용]:
  python scripts/start_local_server.py --backend transformers \\
    --model-path /models/Phi-3-mini --device cpu
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
        help="[transformers] 로컬 모델 디렉토리 경로",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="[transformers] 디바이스 (auto/cpu/cuda, default: auto)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="[transformers] 8-bit 양자화 (~절반 VRAM)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="[transformers] 4-bit 양자화 NF4 (~1/4 VRAM, A100 12GB 이하 권장)",
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

    if args.load_in_4bit and args.load_in_8bit:
        logger.error("--load-in-4bit과 --load-in-8bit은 동시에 사용할 수 없습니다.")
        sys.exit(1)

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
        "load_in_4bit": args.load_in_4bit,
    }

    logger.info(f"백엔드: {args.backend}")
    if args.backend == "transformers":
        if not args.model_path:
            logger.error("--model-path 옵션이 필요합니다.")
            sys.exit(1)
        logger.info(f"모델 경로: {args.model_path}")
        if args.load_in_4bit:
            logger.info("양자화: 4-bit NF4 (최대 메모리 절약)")
        elif args.load_in_8bit:
            logger.info("양자화: 8-bit")
        else:
            logger.info("양자화: 없음 (FP16)")

    app = create_app(backend_name=args.backend, **backend_kwargs)

    logger.info(f"서버 시작: http://{args.host}:{args.port}")
    logger.info("RS-Agent config.yaml의 llm.base_url을 아래로 설정하세요:")
    logger.info(f"  base_url: \"http://localhost:{args.port}/v1\"")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
