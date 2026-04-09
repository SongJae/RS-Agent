#!/usr/bin/env bash
# =============================================================================
# setup_offline.sh  —  폐쇄망 PC에서 실행
#
# rs-agent-env.tar.gz 없이 wheel 파일로 직접 설치합니다.
#
# USB 구조:
#   /usb/
#   ├── RS-Agent/               ← 저장소 (git clone 또는 zip, ~300 MB)
#   │   └── packages/wheels/    ← 코어 패키지 (git 포함, ~104 MB)
#   └── ml-wheels/              ← ML 패키지 whl 파일 (~2.8 GB)
#       ├── torch-*.whl
#       ├── transformers-*.whl
#       └── ...
#
# 사용법:
#   bash RS-Agent/scripts/setup_offline.sh /media/usb/ml-wheels
#   bash RS-Agent/scripts/setup_offline.sh  # ml-wheels 위치 자동 탐색
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MINICONDA_SH="$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh"
CONDA_INSTALL_DIR="$HOME/miniconda3-rs-pack"
ENV_DIR="$HOME/rs-agent-env"
CORE_WHEELS_DIR="$PROJECT_DIR/packages/wheels"

# ML wheels 위치 결정
if [ $# -ge 1 ]; then
  ML_WHEELS_DIR="$1"
else
  # 자동 탐색: USB → 프로젝트 내부 순
  ML_WHEELS_DIR=""
  for candidate in \
    "$(dirname "$PROJECT_DIR")/ml-wheels" \
    "$PROJECT_DIR/packages/ml-wheels" \
    "$PROJECT_DIR/packages-ml-wheels"; do
    if [ -d "$candidate" ] && [ "$(ls -A "$candidate" 2>/dev/null)" ]; then
      ML_WHEELS_DIR="$candidate"
      break
    fi
  done

  # packages-ml.tar.gz 가 있으면 자동 압축 해제
  if [ -z "$ML_WHEELS_DIR" ]; then
    for ml_tar in \
      "$(dirname "$PROJECT_DIR")/packages-ml.tar.gz" \
      "$PROJECT_DIR/packages-ml.tar.gz"; do
      if [ -f "$ml_tar" ]; then
        echo "[준비] $ml_tar 압축 해제 중..."
        tar -xzf "$ml_tar" -C "$PROJECT_DIR"
        ML_WHEELS_DIR="$PROJECT_DIR/packages/ml-wheels"
        echo "  → 완료"
        break
      fi
    done
  fi
fi

echo "======================================================"
echo " RS-Agent 오프라인 직접 설치 (wheel → venv)"
echo " ML wheels  : ${ML_WHEELS_DIR:-없음 (ML 기능 비활성)}"
echo " 코어 wheels: $CORE_WHEELS_DIR"
echo " 설치 경로  : $ENV_DIR"
echo "======================================================"

# ── 1. Python 결정 ───────────────────────────────────────────────────────────
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
    echo "[ERROR] Python 3.11을 찾을 수 없습니다. python3.11 또는 Miniconda가 필요합니다."
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
echo "  → $ENV_DIR"

# ── 3. 코어 패키지 설치 ──────────────────────────────────────────────────────
echo ""
echo "[3/5] 코어 패키지 설치 중 (로컬 wheels)..."
if [ -d "$CORE_WHEELS_DIR" ] && [ "$(ls -A "$CORE_WHEELS_DIR" 2>/dev/null)" ]; then
  "$PIP" install \
    numpy scikit-learn faiss-cpu Pillow PyYAML rank-bm25 rich python-dotenv requests \
    openai anthropic fastapi "uvicorn[standard]" tiktoken \
    --find-links "$CORE_WHEELS_DIR" -q
  echo "  → 완료"
else
  echo "[WARN] packages/wheels/ 없음 — 코어 패키지 설치 건너뜀"
fi

# ── 4. ML 패키지 설치 ────────────────────────────────────────────────────────
echo ""
echo "[4/5] ML 패키지 설치 중..."
if [ -n "$ML_WHEELS_DIR" ] && [ -d "$ML_WHEELS_DIR" ]; then
  "$PIP" install \
    torch transformers accelerate bitsandbytes sentence-transformers gradio \
    --no-index --find-links "$ML_WHEELS_DIR" -q
  echo "  → 완료 (오프라인, $ML_WHEELS_DIR)"
else
  echo "[WARN] ML wheels 없음 — sentence-transformers/torch 없이 실행됩니다"
  echo "       (Mock LLM 서버 + BM25 키워드 검색만 사용 가능)"
fi

# ── 5. 실행 스크립트 생성 ─────────────────────────────────────────────────────
echo ""
echo "[5/5] 실행 스크립트 생성 중..."

cat > "$PROJECT_DIR/run_server.sh" <<RUNEOF
#!/usr/bin/env bash
# LLM 서버 — Mock 모드 (GPU 불필요)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$ENV_DIR/bin:\$PATH"
cd "$PROJECT_DIR"
python scripts/start_local_server.py --backend mock --port 11434
RUNEOF

cat > "$PROJECT_DIR/run_server_gpu.sh" <<RUNEOF
#!/usr/bin/env bash
# LLM 서버 — GPU 모드
# 사용법: bash run_server_gpu.sh models/Qwen2.5-7B-Instruct [--load-in-4bit]
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$ENV_DIR/bin:\$PATH"
cd "$PROJECT_DIR"
python scripts/start_local_server.py --backend transformers \
  --model-path "\${1:?모델 경로가 필요합니다}" \${2:-} --port 11434
RUNEOF

cat > "$PROJECT_DIR/run_ui.sh" <<RUNEOF
#!/usr/bin/env bash
# Gradio Web UI
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$ENV_DIR/bin:\$PATH"
cd "$PROJECT_DIR"
python app.py --port 7860
RUNEOF

chmod +x "$PROJECT_DIR/run_server.sh" \
         "$PROJECT_DIR/run_server_gpu.sh" \
         "$PROJECT_DIR/run_ui.sh"

echo "  → run_server.sh / run_server_gpu.sh / run_ui.sh 생성 완료"

# ── 완료 ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " 설치 완료!"
echo ""
echo " [터미널 1] LLM 서버 (Mock 모드)"
echo "   bash $PROJECT_DIR/run_server.sh"
echo ""
echo " [터미널 1] A100 GPU 서버 (FP16)"
echo "   bash $PROJECT_DIR/run_server_gpu.sh models/Qwen2.5-7B-Instruct"
echo ""
echo " [터미널 1] A100 GPU 서버 (4-bit, ~12 GB VRAM)"
echo "   bash $PROJECT_DIR/run_server_gpu.sh models/Qwen2.5-7B-Instruct --load-in-4bit"
echo ""
echo " [터미널 2] Web UI"
echo "   bash $PROJECT_DIR/run_ui.sh"
echo ""
echo " 브라우저: http://localhost:7860"
echo "======================================================"
