#!/usr/bin/env bash
# =============================================================================
# pack_conda_env.sh  —  인터넷 PC 또는 폐쇄망 PC에서 실행
#
# RS-Agent 실행 환경을 rs-agent-env.tar.gz로 패킹합니다.
#
# [conda 모드] conda가 있으면 conda-pack 사용 (권장)
#   → install_offline.sh 에서 conda-unpack 으로 경로 자동 재설정
#
# [venv 모드] conda 없으면 python -m venv + tar 사용
#   → install_offline.sh 에서 sed 로 경로 수정
#
# 오프라인 설치 조건 (packages/ml-wheels/ 있을 때 인터넷 불필요):
#   bash scripts/pack_conda_env.sh
#
# 옵션:
#   --no-flash-attn   Flash Attention 2 빌드 건너뜀
#   --ml-wheels DIR   ML wheels 디렉토리 지정
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT="$PROJECT_DIR/rs-agent-env.tar.gz"
CORE_WHEELS_DIR="$PROJECT_DIR/packages/wheels"
ML_WHEELS_DIR="$PROJECT_DIR/packages/ml-wheels"
NO_FLASH=0
CUSTOM_ML_WHEELS=""

for arg in "$@"; do
  case $arg in
    --no-flash-attn) NO_FLASH=1 ;;
    --ml-wheels)     shift; CUSTOM_ML_WHEELS="$1" ;;
  esac
done
[ -n "$CUSTOM_ML_WHEELS" ] && ML_WHEELS_DIR="$CUSTOM_ML_WHEELS"

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
  echo " RS-Agent 환경 패킹 [완전 오프라인 모드]"
else
  OFFLINE_MODE=0
  echo "======================================================"
  echo " RS-Agent 환경 패킹 [온라인 모드 — 인터넷 필요]"
fi
echo " 출력 파일  : $OUTPUT"
echo "======================================================"

# ── conda 탐색 ────────────────────────────────────────────────────────────────
CONDA_BIN=""
for candidate in conda "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
                 "/opt/conda/bin/conda" "/usr/local/anaconda3/bin/conda"; do
  if command -v "$candidate" &>/dev/null 2>&1; then
    CONDA_BIN=$(command -v "$candidate" 2>/dev/null || echo "$candidate")
    break
  fi
done

if [ -n "$CONDA_BIN" ]; then
  echo "  conda 감지: $CONDA_BIN"
  _do_conda_mode
else
  echo "  conda 없음 → venv 모드 사용"
  _do_venv_mode
fi

# ── [conda 모드] conda create + conda-pack ────────────────────────────────────
_do_conda_mode() {
  CONDA_BASE=$("$CONDA_BIN" info --base 2>/dev/null || echo "")
  ENV_NAME="rs-agent-$$"

  echo ""
  echo "[1/5] conda 환경 생성 중 (python=3.11)..."
  # --offline 으로 먼저 시도 (로컬 캐시 사용), 실패시 온라인
  if ! "$CONDA_BIN" create -n "$ENV_NAME" python=3.11 -y --offline -q 2>/dev/null; then
    echo "  로컬 캐시에 python=3.11 없음 → 온라인 다운로드..."
    "$CONDA_BIN" create -n "$ENV_NAME" python=3.11 -y -q
  fi
  echo "  → 완료: $ENV_NAME"

  # conda 환경의 pip
  if [ -n "$CONDA_BASE" ] && [ -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
    PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"
  else
    PIP=$("$CONDA_BIN" run -n "$ENV_NAME" which pip 2>/dev/null)
  fi
  "$PIP" install --upgrade pip -q

  echo ""
  echo "[2/5] 코어 패키지 설치 중..."
  if [ -d "$CORE_WHEELS_DIR" ] && [ "$(ls -A "$CORE_WHEELS_DIR" 2>/dev/null)" ]; then
    "$PIP" install \
      numpy scikit-learn faiss-cpu Pillow PyYAML rank-bm25 rich \
      python-dotenv requests openai anthropic fastapi "uvicorn[standard]" tiktoken \
      --find-links "$CORE_WHEELS_DIR" --no-index -q
  else
    "$PIP" install \
      numpy scikit-learn faiss-cpu Pillow PyYAML rank-bm25 rich \
      python-dotenv requests openai anthropic fastapi "uvicorn[standard]" tiktoken -q
  fi

  echo ""
  echo "[3/5] ML 패키지 설치 중..."
  if [ $OFFLINE_MODE -eq 1 ]; then
    "$PIP" install \
      torch transformers accelerate bitsandbytes sentence-transformers gradio \
      --no-index --find-links "$ML_WHEELS_DIR" -q
  else
    "$PIP" install \
      torch "transformers>=4.40.0" accelerate bitsandbytes \
      "sentence-transformers>=2.2.2" "gradio>=6.0.0" -q
    if [ $NO_FLASH -eq 0 ]; then
      "$PIP" install flash-attn --no-build-isolation -q 2>/dev/null || \
        echo "  [WARN] flash-attn 설치 실패 — 건너뜁니다"
    fi
  fi

  echo ""
  echo "[4/5] conda-pack 설치 중..."
  # 로컬 wheels 에서 설치 시도 → 없으면 conda-forge 에서 설치
  "$PIP" install conda-pack \
    --find-links "$CORE_WHEELS_DIR" --no-index -q 2>/dev/null || \
    "$CONDA_BIN" install -n "$ENV_NAME" -c conda-forge conda-pack -y -q 2>/dev/null || \
    "$PIP" install conda-pack -q

  echo ""
  echo "[5/5] conda-pack 으로 패킹 중..."
  "$CONDA_BIN" pack -n "$ENV_NAME" -o "$OUTPUT" --ignore-editable-packages -q
  "$CONDA_BIN" env remove -n "$ENV_NAME" -y -q 2>/dev/null || true

  _print_done "conda-pack"
}

# ── [venv 모드] python -m venv + tar ─────────────────────────────────────────
_do_venv_mode() {
  MINICONDA_SH="$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh"
  CONDA_INSTALL_DIR="$HOME/miniconda3-rs-pack"
  ENV_DIR="$HOME/rs-agent-env"

  echo ""
  echo "[1/5] Python 환경 준비 중..."

  PYTHON_BIN=""
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
      echo "  → Miniconda Python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"
    else
      echo "[ERROR] Python 3.11 또는 Miniconda가 필요합니다."
      exit 1
    fi
  fi

  echo ""
  echo "[2/5] Python venv 생성 중..."
  [ -d "$ENV_DIR" ] && rm -rf "$ENV_DIR"
  "$PYTHON_BIN" -m venv "$ENV_DIR"
  PIP="$ENV_DIR/bin/pip"
  "$PIP" install --upgrade pip -q
  # 원본 경로 저장 (install_offline.sh 에서 sed 치환용)
  echo "$ENV_DIR" > "$ENV_DIR/.rs_original_path"
  echo "  → $ENV_DIR"

  echo ""
  echo "[3/5] 코어 패키지 설치 중..."
  if [ -d "$CORE_WHEELS_DIR" ] && [ "$(ls -A "$CORE_WHEELS_DIR" 2>/dev/null)" ]; then
    "$PIP" install \
      numpy scikit-learn faiss-cpu Pillow PyYAML rank-bm25 rich \
      python-dotenv requests openai anthropic fastapi "uvicorn[standard]" tiktoken \
      --find-links "$CORE_WHEELS_DIR" -q
  else
    "$PIP" install \
      numpy scikit-learn faiss-cpu Pillow PyYAML rank-bm25 rich \
      python-dotenv requests openai anthropic fastapi "uvicorn[standard]" tiktoken -q
  fi

  echo ""
  echo "[4/5] ML 패키지 설치 중..."
  if [ $OFFLINE_MODE -eq 1 ]; then
    "$PIP" install \
      torch transformers accelerate bitsandbytes sentence-transformers gradio \
      --no-index --find-links "$ML_WHEELS_DIR" -q
  else
    "$PIP" install \
      torch "transformers>=4.40.0" accelerate bitsandbytes \
      "sentence-transformers>=2.2.2" "gradio>=6.0.0" -q
    if [ $NO_FLASH -eq 0 ]; then
      "$PIP" install flash-attn --no-build-isolation -q 2>/dev/null || \
        echo "  [WARN] flash-attn 설치 실패 — 건너뜁니다"
    fi
  fi

  echo ""
  echo "[5/5] venv tar.gz 패킹 중..."
  tar -czf "$OUTPUT" -C "$(dirname "$ENV_DIR")" "$(basename "$ENV_DIR")"

  _print_done "venv+tar"
}

# ── 완료 메시지 ───────────────────────────────────────────────────────────────
_print_done() {
  MODE="${1:-}"
  SIZE=$(du -sh "$OUTPUT" 2>/dev/null | cut -f1)
  echo ""
  echo "======================================================"
  echo " 완료: $OUTPUT ($SIZE)  [모드: $MODE]"
  echo ""
  echo " 폐쇄망 PC 설치:"
  echo "   bash RS-Agent/scripts/install_offline.sh ~/rs-agent-env.tar.gz"
  echo "======================================================"
}
