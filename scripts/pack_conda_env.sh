#!/usr/bin/env bash
# =============================================================================
# pack_conda_env.sh  —  인터넷 연결 PC에서 실행
#
# conda-pack을 사용해 RS-Agent 실행 환경 전체를 하나의 tar.gz로 묶습니다.
# 생성된 파일을 USB/HDD로 폐쇄망 PC에 복사하면 pip/conda 없이 바로 실행됩니다.
#
# 사용법:
#   bash scripts/pack_conda_env.sh
#   bash scripts/pack_conda_env.sh --cpu-only     # GPU 없는 환경 (CPU-only PyTorch)
#   bash scripts/pack_conda_env.sh --no-flash-attn # flash-attn 제외 (빌드 오래 걸릴 때)
#
# 결과물:
#   rs-agent-env.tar.gz  (~4–8 GB, 이식 가능한 conda 환경)
# =============================================================================
set -euo pipefail

ENV_NAME="rs-agent"
OUTPUT="rs-agent-env.tar.gz"
CPU_ONLY=0
NO_FLASH=0

for arg in "$@"; do
  case $arg in
    --cpu-only)     CPU_ONLY=1 ;;
    --no-flash-attn) NO_FLASH=1 ;;
  esac
done

echo "======================================================"
echo " RS-Agent 오프라인 패키지 생성"
echo " 환경 이름  : $ENV_NAME"
echo " 출력 파일  : $OUTPUT"
echo " CPU only   : $CPU_ONLY"
echo "======================================================"

# ── 1. Miniconda/Conda 확인 ───────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
  echo "[ERROR] conda가 없습니다. Miniconda를 먼저 설치하세요."
  echo "  https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

# ── 2. 기존 환경 제거 후 재생성 ──────────────────────────────────────────────
echo ""
echo "[1/5] conda 환경 생성 중 ($ENV_NAME, Python 3.10)..."
conda env remove -n "$ENV_NAME" --yes 2>/dev/null || true
conda create -n "$ENV_NAME" python=3.10 -y

# ── 3. 패키지 설치 ───────────────────────────────────────────────────────────
echo ""
echo "[2/5] 패키지 설치 중..."

# conda activate 대신 직접 경로 사용 (스크립트 내 호환성)
CONDA_BASE=$(conda info --base)
PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"

# PyTorch 설치
if [ $CPU_ONLY -eq 1 ]; then
  echo "  → CPU-only PyTorch 설치"
  "$PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  echo "  → CUDA 12.1 PyTorch 설치 (A100용)"
  "$PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# Core 패키지
"$PIP" install \
  numpy>=1.24.0 \
  scikit-learn>=1.3.0 \
  faiss-cpu>=1.7.4 \
  sentence-transformers>=2.2.2 \
  Pillow>=10.0.0 \
  PyYAML>=6.0 \
  rank-bm25>=0.2.2 \
  rich>=13.0.0 \
  python-dotenv>=1.0.0 \
  requests>=2.31.0

# UI
"$PIP" install "gradio>=6.0.0"

# LLM 관련
"$PIP" install \
  "openai>=1.0.0" \
  "anthropic>=0.40.0" \
  "transformers>=4.40.0" \
  "accelerate>=0.27.0" \
  "bitsandbytes>=0.43.0" \
  "fastapi>=0.110.0" \
  "uvicorn[standard]>=0.29.0" \
  tiktoken

# Flash Attention 2 (A100 전용, 빌드 시간 ~10분)
if [ $NO_FLASH -eq 0 ] && [ $CPU_ONLY -eq 0 ]; then
  echo "  → Flash Attention 2 설치 (A100 최적화, 시간 소요)..."
  "$PIP" install flash-attn --no-build-isolation || echo "  [WARN] flash-attn 설치 실패 — 건너뜁니다 (선택 사항)"
fi

# conda-pack 설치
conda install -n "$ENV_NAME" conda-pack -c conda-forge -y

# ── 4. 환경 패킹 ─────────────────────────────────────────────────────────────
echo ""
echo "[3/5] 환경 패킹 중 (시간 소요)..."
conda pack -n "$ENV_NAME" -o "$OUTPUT" --ignore-missing-files

echo ""
echo "[4/5] 완료 확인..."
SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "  → $OUTPUT ($SIZE)"

# ── 5. 전송 안내 ─────────────────────────────────────────────────────────────
echo ""
echo "[5/5] 폐쇄망 전송 준비 목록:"
echo "  ① $OUTPUT                        ← conda 환경 전체"
echo "  ② RS-Agent/ 프로젝트 디렉토리     ← 코드 + 모델 + 데이터"
echo ""
echo "  USB/HDD에 다음 2개를 복사하세요:"
echo "    /path/to/usb/"
echo "    ├── rs-agent-env.tar.gz"
echo "    └── RS-Agent/"
echo ""
echo "  폐쇄망 PC에서 실행:"
echo "    bash RS-Agent/scripts/install_offline.sh"
echo "======================================================"
