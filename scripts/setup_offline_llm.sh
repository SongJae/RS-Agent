#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# RS-Agent 폐쇄망 LLM 설정 가이드
# ══════════════════════════════════════════════════════════════
#
# 이 스크립트는 폐쇄망 환경에서 LLM을 설정하는 방법을 안내합니다.
# 실제 실행 전에 내용을 확인하고 환경에 맞게 수정하세요.
#
# 방법 1: RS-Agent 내장 로컬 서버 (추천)
# 방법 2: Ollama (오프라인 패키지 반입)
# 방법 3: vLLM (GPU 서버)
# ══════════════════════════════════════════════════════════════

set -e

echo "================================================"
echo "  RS-Agent 폐쇄망 LLM 설정 가이드"
echo "================================================"
echo ""

# ── 방법 1: RS-Agent 내장 로컬 서버 ──────────────────────────
setup_builtin_server() {
    echo "[방법 1] RS-Agent 내장 로컬 서버 (fastapi + transformers)"
    echo ""
    echo "1) 인터넷 환경에서 모델 다운로드:"
    echo "   python scripts/download_model.py \\"
    echo "     --model Qwen/Qwen2.5-7B-Instruct \\"
    echo "     --output /models/Qwen2.5-7B-Instruct"
    echo ""
    echo "2) 다운로드된 /models/ 디렉토리를 폐쇄망 서버에 복사"
    echo ""
    echo "3) 폐쇄망에서 서버 실행:"
    echo "   python scripts/start_local_server.py \\"
    echo "     --backend transformers \\"
    echo "     --model-path /models/Qwen2.5-7B-Instruct \\"
    echo "     --port 11434"
    echo ""
    echo "4) config.yaml 설정:"
    echo "   llm:"
    echo "     backend: openai_compatible"
    echo "     base_url: http://localhost:11434/v1"
    echo "     model: Qwen2.5-7B-Instruct"
    echo ""
}

# ── 방법 2: Ollama 오프라인 설치 ─────────────────────────────
setup_ollama_offline() {
    echo "[방법 2] Ollama 오프라인 설치"
    echo ""
    echo "1) 인터넷 환경에서 Ollama 설치 바이너리 다운로드:"
    echo "   # Linux x86_64"
    echo "   curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64 \\"
    echo "     -o ollama"
    echo "   chmod +x ollama"
    echo ""
    echo "2) 모델 GGUF 파일 다운로드 (huggingface.co):"
    echo "   # 예: Qwen2.5-7B-Instruct Q4_K_M"
    echo "   # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF"
    echo ""
    echo "3) 폐쇄망에서 Ollama + GGUF 복사 후 모델 임포트:"
    echo "   ./ollama serve &"
    echo "   cat > Modelfile << 'EOF'"
    echo "   FROM /path/to/model.gguf"
    echo "   PARAMETER temperature 0.1"
    echo "   PARAMETER num_ctx 4096"
    echo "   EOF"
    echo "   ./ollama create rs-model -f Modelfile"
    echo "   ./ollama run rs-model  # 동작 확인"
    echo ""
    echo "4) config.yaml 설정:"
    echo "   llm:"
    echo "     backend: ollama"
    echo "     base_url: http://localhost:11434/v1"
    echo "     model: rs-model"
    echo ""
}

# ── 방법 3: vLLM GPU 서버 ────────────────────────────────────
setup_vllm_offline() {
    echo "[방법 3] vLLM (GPU 서버 권장)"
    echo ""
    echo "1) 인터넷 환경에서 vLLM 및 모델 준비:"
    echo "   pip download vllm -d /packages/vllm"
    echo "   python scripts/download_model.py \\"
    echo "     --model Qwen/Qwen2.5-7B-Instruct \\"
    echo "     --output /models/Qwen2.5-7B-Instruct"
    echo ""
    echo "2) 폐쇄망에서 vLLM 설치:"
    echo "   pip install --no-index --find-links=/packages/vllm vllm"
    echo ""
    echo "3) 폐쇄망에서 vLLM 서버 실행:"
    echo "   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\"
    echo "   python -m vllm.entrypoints.openai.api_server \\"
    echo "     --model /models/Qwen2.5-7B-Instruct \\"
    echo "     --host 0.0.0.0 \\"
    echo "     --port 11434 \\"
    echo "     --tensor-parallel-size 1"
    echo ""
    echo "4) config.yaml 설정:"
    echo "   llm:"
    echo "     backend: openai_compatible"
    echo "     base_url: http://localhost:11434/v1"
    echo "     model: /models/Qwen2.5-7B-Instruct"
    echo ""
}

# ── 테스트용 Mock 서버 ────────────────────────────────────────
setup_mock_server() {
    echo "[테스트] Mock 서버 (모델 없이 파이프라인 검증)"
    echo ""
    echo "   python scripts/start_local_server.py --backend mock --port 11434"
    echo ""
    echo "   config.yaml:"
    echo "   llm:"
    echo "     backend: openai_compatible"
    echo "     base_url: http://localhost:11434/v1"
    echo "     model: mock-rs-agent"
    echo ""
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
setup_builtin_server
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
setup_ollama_offline
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
setup_vllm_offline
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
setup_mock_server
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "공통 환경변수 (.env 파일):"
echo "  HF_HUB_OFFLINE=1"
echo "  TRANSFORMERS_OFFLINE=1"
echo "  LLM_BASE_URL=http://localhost:11434/v1"
echo "  LLM_API_KEY=local"
