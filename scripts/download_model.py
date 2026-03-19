#!/usr/bin/env python3
"""
모델 사전 다운로드 스크립트 (인터넷 환경에서 실행)

폐쇄망으로 반입할 모델을 로컬 경로에 저장합니다.
저장된 디렉토리를 폐쇄망 서버에 복사하여 사용하세요.

사용 예:
  # LLM 모델 다운로드
  python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct --output /models/Qwen2.5-7B-Instruct
  python scripts/download_model.py --model meta-llama/Llama-3.1-8B-Instruct --output /models/Llama-3.1-8B-Instruct

  # 임베딩 모델 다운로드
  python scripts/download_model.py --model sentence-transformers/all-MiniLM-L6-v2 --output /models/all-MiniLM-L6-v2 --type embedding
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="폐쇄망 반입용 모델 다운로드")
    parser.add_argument("--model", required=True, help="HuggingFace 모델 ID")
    parser.add_argument("--output", required=True, help="저장할 로컬 경로")
    parser.add_argument(
        "--type",
        choices=["llm", "embedding"],
        default="llm",
        help="모델 종류 (default: llm)",
    )
    return parser.parse_args()


def download_llm(model_id: str, output_path: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"[LLM] 다운로드: {model_id} → {output_path}")
    os.makedirs(output_path, exist_ok=True)

    print("  토크나이저 다운로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("  모델 가중치 다운로드 중 (시간이 걸릴 수 있습니다)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.save_pretrained(output_path)
    print(f"  완료: {output_path}")


def download_embedding(model_id: str, output_path: str):
    from sentence_transformers import SentenceTransformer

    print(f"[Embedding] 다운로드: {model_id} → {output_path}")
    model = SentenceTransformer(model_id)
    model.save(output_path)
    print(f"  완료: {output_path}")


def main():
    args = parse_args()
    if args.type == "embedding":
        download_embedding(args.model, args.output)
    else:
        download_llm(args.model, args.output)

    print("\n모델 준비 완료!")
    print(f"폐쇄망 서버에서 사용할 설정:")
    if args.type == "embedding":
        print(f"  config.yaml > knowledge_space.local_model_path: \"{args.output}\"")
        print(f"  config.yaml > solution_space.local_model_path: \"{args.output}\"")
    else:
        print(f"  python scripts/start_local_server.py --backend transformers --model-path \"{args.output}\"")
        print(f"  config.yaml > llm.base_url: \"http://localhost:11434/v1\"")


if __name__ == "__main__":
    main()
