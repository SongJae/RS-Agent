#!/usr/bin/env bash
# =============================================================================
# pack_conda_env.sh  —  인터넷 연결 PC 또는 폐쇄망 PC에서 실행
#
# RS-Agent 실행 환경 전체를 rs-agent-env.tar.gz로 묶습니다.
# conda 채널(repo.anaconda.com) 접근 없이 venv + pip 만으로 환경을 구성합니다.
#
# [완전 오프라인 모드] packages/ml-wheels/ 가 있으면 인터넷 불필요:
#   USB 구조:
#     /usb/
#     ├── RS-Agent/              ← 저장소 (Miniconda 분할파일 + core wheels 포함)
#     └── packages-ml.tar.gz    ← ML wheels (~2.8 GB, scripts/download_ml.sh 로 생성)
#
#   실행:
#     tar -xzf packages-ml.tar.gz -C RS-Agent/
#     bash RS-Agent/scripts/pack_conda_env.sh     # 완전 오프라인
#
# [온라인 모드] packages/ml-wheels/ 없으면 인터넷에서 자동 다운로드
#
# 사용법:
#   bash scripts/pack_conda_env.sh
#   bash scripts/pack_conda_env.sh --no-flash-attn
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MINICONDA_SH="$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh"
CONDA_INSTALL_DIR="$HOME/miniconda3-rs-pack"
ENV_DIR="$HOME/rs-agent-env"
OUTPUT="$PROJECT_DIR/rs-agent-env.tar.gz"
ML_WHEELS_DIR="$PROJECT_DIR/packages/ml-wheels"
CORE_WHEELS_DIR="$PROJECT_DIR/packages/wheels"
NO_FLASH=0

for arg in "$@"; do
  case $arg in
    --no-flash-attn) NO_FLASH=1 ;;
  esac
done

# packages-ml.tar.gz 가 있고 ml-wheels 가 없으면 자동 압축 해제
for ML_TAR in "$PROJECT_DIR/packages-ml.tar.gz" "$(dirname "$PROJECT_DIR")/packages-ml.tar.gz"; do
  if [ ! -d "$ML_WHEELS_DIR" ] && [ -f "$ML_TAR" ]; then
    echo "[준비] $ML_TAR 압축 해제 중..."
    tar -xzf "$ML_TAR" -C "$PROJECT_DIR"
    echo "  → 완료"
    break
  fi
done

# 오프라인/온라인 모드 결정
if [ -d "$ML_WHEELS_DIR" ] && [ "$(ls -A "$ML_WHEELS_DIR" 2>/dev/null)" ]; then
  OFFLINE_MODE=1
  echo "======================================================"
  echo " RS-Agent 패키지 생성 [완전 오프라인 모드]"
else
  OFFLINE_MODE=0
  echo "======================================================"
  echo " RS-Agent 패키지 생성 [온라인 모드 — 인터넷 필요]"
fi
echo " 출력 파일  : $OUTPUT"
echo "======================================================"

# ── 1. Python 결정 (시스템 Python 우선, 없으면 Miniconda) ────────────────────
echo ""
echo "[1/5] Python 환경 준비 중..."

PYTHON_BIN=""

# 시스템 Python 3.11 확인 (cp311 wheels와 호환)
for candidate in python3.11 python3 python; do
  if command -v "$candidate" &>/dev/null; then
    VER=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    if [ "$VER" = "3.11" ]; then
      PYTHON_BIN=$(command -v "$candidate")
      echo "  → 시스템 Python 3.11 사용: $PYTHON_BIN"
      break
    fi
  fi
done

# 시스템에 Python 3.11 없으면 Miniconda 설치
if [ -z "$PYTHON_BIN" ]; then
  PART_AA="$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh.partaa"
  if [ ! -f "$MINICONDA_SH" ] && [ -f "$PART_AA" ]; then
    echo "  분할 파일 복원 중..."
    cat "$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh.part"* > "$MINICONDA_SH"
    chmod +x "$MINICONDA_SH"
  fi
  if [ -f "$MINICONDA_SH" ]; then
    echo "  Miniconda 설치 중..."
    bash "$MINICONDA_SH" -b -p "$CONDA_INSTALL_DIR" -u
    PYTHON_BIN="$CONDA_INSTALL_DIR/bin/python"
    echo "  → Miniconda Python 사용: $PYTHON_BIN ($("$PYTHON_BIN" --version))"
  else
    echo "[ERROR] Python을 찾을 수 없습니다. python3.11 또는 Miniconda가 필요합니다."
    exit 1
  fi
fi

# ── 2. venv 생성 ─────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Python venv 생성 중..."
[ -d "$ENV_DIR" ] && rm -rf "$ENV_DIR"
"$PYTHON_BIN" -m venv "$ENV_DIR"
PIP="$ENV_DIR/bin/pip"
"$PIP" install --upgrade pip -q
echo "$ENV_DIR" > "$ENV_DIR/.rs_original_path"
echo "  → 생성 완료: $ENV_DIR"

# ── 3. 패키지 설치 ───────────────────────────────────────────────────────────
echo ""
echo "[3/5] 패키지 설치 중..."

echo "  → 코어 패키지 (로컬 wheels)..."
"$PIP" install \
  numpy scikit-learn faiss-cpu Pillow PyYAML rank-bm25 rich python-dotenv requests \
  openai anthropic fastapi "uvicorn[standard]" tiktoken \
  --find-links "$CORE_WHEELS_DIR" -q

if [ $OFFLINE_MODE -eq 1 ]; then
  echo "  → ML 패키지 (로컬 ml-wheels, 인터넷 불필요)..."
  "$PIP" install \
    torch transformers accelerate bitsandbytes sentence-transformers gradio \
    --no-index --find-links "$ML_WHEELS_DIR" -q
else
  echo "  → ML 패키지 (인터넷 다운로드)..."
  "$PIP" install \
    torch "transformers>=4.40.0" accelerate bitsandbytes \
    "sentence-transformers>=2.2.2" "gradio>=6.0.0" -q

  if [ $NO_FLASH -eq 0 ]; then
    echo "  → Flash Attention 2 (선택)..."
    "$PIP" install flash-attn --no-build-isolation -q 2>/dev/null || \
      echo "  [WARN] flash-attn 설치 실패 — 건너뜁니다"
  fi
fi

# ── 4. tar.gz 패킹 ───────────────────────────────────────────────────────────
echo ""
echo "[4/5] 환경 패킹 중..."
tar -czf "$OUTPUT" -C "$(dirname "$ENV_DIR")" "$(basename "$ENV_DIR")"

echo ""
echo "[5/5] 완료 확인..."
SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "  → $OUTPUT ($SIZE)"
echo ""
echo "======================================================"
echo " USB에 복사: rs-agent-env.tar.gz ($SIZE) + RS-Agent/"
echo " 폐쇄망 PC: bash RS-Agent/scripts/install_offline.sh"
echo "======================================================"
