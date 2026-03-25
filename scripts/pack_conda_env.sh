#!/usr/bin/env bash
# =============================================================================
# pack_conda_env.sh  —  인터넷 연결 PC에서 실행
#
# Miniconda를 설치하고 RS-Agent 실행 환경 전체를 rs-agent-env.tar.gz로 묶습니다.
# 생성된 파일을 USB/HDD로 폐쇄망 PC에 복사하면 바로 실행 가능합니다.
#
# 사용법:
#   bash scripts/pack_conda_env.sh
#   bash scripts/pack_conda_env.sh --no-flash-attn   # flash-attn 제외 (빌드 생략)
#
# 저장소에 포함된 파일:
#   install/Miniconda3-latest-Linux-x86_64.sh  ← Miniconda 설치 파일 (자동 사용)
#   packages/wheels/                            ← 코어 패키지 wheels (자동 사용)
#
# 결과물:
#   rs-agent-env.tar.gz  (~3 GB, git 제외 — USB로 별도 복사)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MINICONDA_SH="$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh"
CONDA_INSTALL_DIR="$HOME/miniconda3-rs"
ENV_DIR="$CONDA_INSTALL_DIR/envs/rs-agent"
OUTPUT="$PROJECT_DIR/rs-agent-env.tar.gz"
NO_FLASH=0

for arg in "$@"; do
  case $arg in
    --no-flash-attn) NO_FLASH=1 ;;
  esac
done

echo "======================================================"
echo " RS-Agent 오프라인 패키지 생성"
echo " Miniconda  : $MINICONDA_SH"
echo " 환경 경로  : $ENV_DIR"
echo " 출력 파일  : $OUTPUT"
echo "======================================================"

# ── 1. Miniconda 설치 ────────────────────────────────────────────────────────
echo ""
echo "[1/5] Miniconda 설치 중..."

if [ ! -f "$MINICONDA_SH" ]; then
  # 분할 파일이 있으면 자동 복원
  PART_AA="$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh.partaa"
  if [ -f "$PART_AA" ]; then
    echo "  분할 파일 복원 중..."
    cat "$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh.part"* > "$MINICONDA_SH"
    chmod +x "$MINICONDA_SH"
    echo "  → 복원 완료: $MINICONDA_SH"
  else
    echo "[ERROR] Miniconda 설치 파일을 찾을 수 없습니다: $MINICONDA_SH"
    echo "  저장소에 install/Miniconda3-latest-Linux-x86_64.sh.partaa/ab 가 있어야 합니다."
    exit 1
  fi
fi

bash "$MINICONDA_SH" -b -p "$CONDA_INSTALL_DIR" -u
CONDA="$CONDA_INSTALL_DIR/bin/conda"

# Terms of Service 동의 (자동)
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "  → Miniconda 설치 완료: $CONDA_INSTALL_DIR"

# ── 2. conda 환경 생성 ───────────────────────────────────────────────────────
echo ""
echo "[2/5] Python 3.10 환경 생성 중..."
"$CONDA" create -p "$ENV_DIR" python=3.10 -y
PIP="$ENV_DIR/bin/pip"

# ── 3. 패키지 설치 ───────────────────────────────────────────────────────────
echo ""
echo "[3/5] 패키지 설치 중..."
WHEELS_DIR="$PROJECT_DIR/packages/wheels"

# 코어 패키지: 저장소 내 wheels에서 먼저 설치 (오프라인 가능)
echo "  → 코어 패키지 (로컬 wheels)..."
"$PIP" install \
  numpy scikit-learn faiss-cpu Pillow PyYAML rank-bm25 rich python-dotenv requests \
  openai anthropic fastapi "uvicorn[standard]" tiktoken \
  --no-index --find-links "$WHEELS_DIR" \
  --no-deps -q 2>/dev/null || \
"$PIP" install \
  numpy scikit-learn faiss-cpu Pillow PyYAML rank-bm25 rich python-dotenv requests \
  openai anthropic fastapi "uvicorn[standard]" tiktoken \
  --find-links "$WHEELS_DIR" -q

# ML 패키지: 인터넷 필요 (torch, transformers, gradio, sentence-transformers)
echo "  → ML 패키지 (인터넷 다운로드)..."
"$PIP" install \
  torch \
  transformers>=4.40.0 accelerate bitsandbytes \
  sentence-transformers>=2.2.2 \
  gradio>=6.0.0 \
  -q

# conda-pack
"$PIP" install conda-pack -q

# Flash Attention 2 (A100 선택)
if [ $NO_FLASH -eq 0 ]; then
  echo "  → Flash Attention 2 (선택 사항, 오래 걸릴 수 있음)..."
  "$PIP" install flash-attn --no-build-isolation -q 2>/dev/null || \
    echo "  [WARN] flash-attn 설치 실패 — 건너뜁니다"
fi

# ── 4. 환경 패킹 ─────────────────────────────────────────────────────────────
echo ""
echo "[4/5] 환경 패킹 중 (시간 소요)..."
"$ENV_DIR/bin/conda-pack" -p "$ENV_DIR" -o "$OUTPUT" --ignore-missing-files

echo ""
echo "[5/5] 완료 확인..."
SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "  → $OUTPUT ($SIZE)"

# ── 전송 안내 ─────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " USB/HDD에 다음을 복사하세요:"
echo ""
echo "   /usb/"
echo "   ├── rs-agent-env.tar.gz      ← 지금 생성된 파일 ($SIZE)"
echo "   └── RS-Agent/                ← 이 저장소"
echo ""
echo " 폐쇄망 PC에서:"
echo "   bash RS-Agent/scripts/install_offline.sh"
echo "======================================================"
