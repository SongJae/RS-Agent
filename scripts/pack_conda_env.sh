#!/usr/bin/env bash
# =============================================================================
# pack_conda_env.sh  —  인터넷 연결 PC에서 실행
#
# Miniconda를 설치하고 RS-Agent 실행 환경 전체를 rs-agent-env.tar.gz로 묶습니다.
# conda 채널(repo.anaconda.com) 접근 없이 venv + pip 만으로 환경을 구성합니다.
#
# 사용법:
#   bash scripts/pack_conda_env.sh
#   bash scripts/pack_conda_env.sh --no-flash-attn   # flash-attn 제외
#
# 결과물:
#   rs-agent-env.tar.gz  (~3 GB, git 제외 — USB로 별도 복사)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MINICONDA_SH="$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh"
CONDA_INSTALL_DIR="$HOME/miniconda3-rs-pack"
ENV_DIR="$HOME/rs-agent-env"
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

# ── 1. Miniconda 분할 파일 복원 ──────────────────────────────────────────────
echo ""
echo "[1/5] Miniconda 준비 중..."

PART_AA="$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh.partaa"
if [ ! -f "$MINICONDA_SH" ]; then
  if [ -f "$PART_AA" ]; then
    echo "  분할 파일 복원 중..."
    cat "$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh.part"* > "$MINICONDA_SH"
    chmod +x "$MINICONDA_SH"
    echo "  → 복원 완료: $MINICONDA_SH"
  else
    echo "[ERROR] Miniconda 설치 파일이 없습니다: $MINICONDA_SH"
    exit 1
  fi
fi

# Miniconda 설치 (베이스 Python만 필요 — conda 채널 불필요)
echo "  Miniconda 설치 중..."
bash "$MINICONDA_SH" -b -p "$CONDA_INSTALL_DIR" -u

# ── 2. venv 생성 (conda create 대신 — 채널 접근 없음) ────────────────────────
echo ""
echo "[2/5] Python venv 생성 중 (conda 채널 불필요)..."
[ -d "$ENV_DIR" ] && rm -rf "$ENV_DIR"
"$CONDA_INSTALL_DIR/bin/python" -m venv "$ENV_DIR"

PIP="$ENV_DIR/bin/pip"
"$PIP" install --upgrade pip -q

# 원본 경로 저장 (install_offline.sh에서 경로 재설정에 사용)
echo "$ENV_DIR" > "$ENV_DIR/.rs_original_path"
echo "  → venv 생성 완료: $ENV_DIR"

# ── 3. 패키지 설치 ───────────────────────────────────────────────────────────
echo ""
echo "[3/5] 패키지 설치 중..."
WHEELS_DIR="$PROJECT_DIR/packages/wheels"

# 코어 패키지: 로컬 wheels (인터넷 불필요)
echo "  → 코어 패키지 (로컬 wheels)..."
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

# Flash Attention 2 (A100 전용, 선택)
if [ $NO_FLASH -eq 0 ]; then
  echo "  → Flash Attention 2 (선택, 오래 걸릴 수 있음)..."
  "$PIP" install flash-attn --no-build-isolation -q 2>/dev/null || \
    echo "  [WARN] flash-attn 설치 실패 — 건너뜁니다"
fi

# ── 4. tar.gz 패킹 ───────────────────────────────────────────────────────────
echo ""
echo "[4/5] 환경 패킹 중..."
ENV_PARENT="$(dirname "$ENV_DIR")"
ENV_BASENAME="$(basename "$ENV_DIR")"
tar -czf "$OUTPUT" -C "$ENV_PARENT" "$ENV_BASENAME"

echo ""
echo "[5/5] 완료 확인..."
SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "  → $OUTPUT ($SIZE)"

echo ""
echo "======================================================"
echo " USB/HDD에 다음을 복사하세요:"
echo ""
echo "   /usb/"
echo "   ├── rs-agent-env.tar.gz      ($SIZE)"
echo "   └── RS-Agent/"
echo ""
echo " 폐쇄망 PC에서:"
echo "   bash RS-Agent/scripts/install_offline.sh"
echo "======================================================"
