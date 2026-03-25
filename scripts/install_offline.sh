#!/usr/bin/env bash
# =============================================================================
# install_offline.sh  —  폐쇄망 PC에서 실행
#
# rs-agent-env.tar.gz를 압축 해제하고 RS-Agent를 바로 실행할 수 있게 설정합니다.
# 인터넷 연결이 전혀 없어도 됩니다.
#
# USB/HDD 구조:
#   /usb/
#   ├── rs-agent-env.tar.gz      ← conda 환경 전체 (~3 GB)
#   └── RS-Agent/                ← 저장소 (코드 + 모델 + wheels + Miniconda 포함)
#       ├── install/
#       │   └── Miniconda3-latest-Linux-x86_64.sh
#       ├── packages/wheels/     ← 코어 pip 패키지
#       ├── models/all-MiniLM-L6-v2/
#       └── scripts/install_offline.sh  ← 이 파일
#
# 사용법:
#   bash RS-Agent/scripts/install_offline.sh
#   bash RS-Agent/scripts/install_offline.sh /media/usb/rs-agent-env.tar.gz
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# tar.gz 위치: 인수로 지정하거나 RS-Agent/ 상위 디렉토리에서 자동 탐색
if [ $# -ge 1 ]; then
  ENV_TAR="$1"
else
  ENV_TAR="$(dirname "$PROJECT_DIR")/rs-agent-env.tar.gz"
  # RS-Agent/ 자체에도 있으면 그쪽 사용
  [ -f "$PROJECT_DIR/rs-agent-env.tar.gz" ] && ENV_TAR="$PROJECT_DIR/rs-agent-env.tar.gz"
fi

INSTALL_DIR="$HOME/rs-agent-env"
MINICONDA_SH="$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh"

echo "======================================================"
echo " RS-Agent 오프라인 설치"
echo " conda 환경 : $ENV_TAR"
echo " 설치 경로  : $INSTALL_DIR"
echo " 프로젝트   : $PROJECT_DIR"
echo "======================================================"

# ── 0. Miniconda 분할 파일 복원 (필요 시) ────────────────────────────────────
PART_AA="$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh.partaa"
if [ ! -f "$MINICONDA_SH" ] && [ -f "$PART_AA" ]; then
  echo ""
  echo "[0/4] Miniconda 분할 파일 복원 중..."
  cat "$PROJECT_DIR/install/Miniconda3-latest-Linux-x86_64.sh.part"* > "$MINICONDA_SH"
  chmod +x "$MINICONDA_SH"
  echo "  → 복원 완료"
fi
echo "======================================================"

# ── 1. tar.gz 확인 ───────────────────────────────────────────────────────────
if [ ! -f "$ENV_TAR" ]; then
  echo "[ERROR] conda 환경 파일을 찾을 수 없습니다: $ENV_TAR"
  echo ""
  echo "  확인 사항:"
  echo "    1) rs-agent-env.tar.gz를 RS-Agent/ 또는 상위 디렉토리에 복사"
  echo "    2) bash install_offline.sh /경로/rs-agent-env.tar.gz"
  exit 1
fi

# ── 2. conda 환경 압축 해제 ──────────────────────────────────────────────────
echo ""
echo "[1/4] conda 환경 압축 해제 중 (시간 소요)..."
[ -d "$INSTALL_DIR" ] && rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
tar -xzf "$ENV_TAR" -C "$INSTALL_DIR"
echo "  → 압축 해제 완료"

# ── 3. 환경 경로 재설정 (conda-unpack) ───────────────────────────────────────
echo ""
echo "[2/4] 환경 경로 재설정 중..."
source "$INSTALL_DIR/bin/activate"
conda-unpack
echo "  → 재설정 완료"

# ── 4. 코어 패키지 보완 (wheels에서 설치) ────────────────────────────────────
WHEELS_DIR="$PROJECT_DIR/packages/wheels"
if [ -d "$WHEELS_DIR" ] && [ "$(ls -A "$WHEELS_DIR")" ]; then
  echo ""
  echo "[3/4] 코어 패키지 보완 설치 (로컬 wheels)..."
  "$INSTALL_DIR/bin/pip" install \
    numpy scikit-learn faiss-cpu Pillow PyYAML rank-bm25 rich python-dotenv requests \
    openai anthropic fastapi "uvicorn[standard]" tiktoken \
    --no-index --find-links "$WHEELS_DIR" -q 2>/dev/null || true
  echo "  → 완료"
else
  echo "[3/4] wheels 디렉토리 없음 — 건너뜀"
fi

# ── 5. 실행 스크립트 생성 ─────────────────────────────────────────────────────
echo ""
echo "[4/4] 실행 스크립트 생성 중..."

cat > "$PROJECT_DIR/run_server.sh" <<RUNEOF
#!/usr/bin/env bash
# LLM 서버 — Mock 모드 (GPU 불필요)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source "$INSTALL_DIR/bin/activate"
cd "$PROJECT_DIR"
python scripts/start_local_server.py --backend mock --port 11434
RUNEOF

cat > "$PROJECT_DIR/run_server_gpu.sh" <<RUNEOF
#!/usr/bin/env bash
# LLM 서버 — GPU 모드 (실제 LLM 추론)
# 사용법: bash run_server_gpu.sh /models/Qwen2.5-7B-Instruct [--load-in-4bit]
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source "$INSTALL_DIR/bin/activate"
cd "$PROJECT_DIR"
python scripts/start_local_server.py --backend transformers \
  --model-path "\${1:?모델 경로가 필요합니다}" \${2:-} --port 11434
RUNEOF

cat > "$PROJECT_DIR/run_ui.sh" <<RUNEOF
#!/usr/bin/env bash
# Gradio Web UI
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source "$INSTALL_DIR/bin/activate"
cd "$PROJECT_DIR"
python app.py --port 7860
RUNEOF

chmod +x "$PROJECT_DIR/run_server.sh" \
         "$PROJECT_DIR/run_server_gpu.sh" \
         "$PROJECT_DIR/run_ui.sh"

echo "  → run_server.sh      생성 완료"
echo "  → run_server_gpu.sh  생성 완료"
echo "  → run_ui.sh          생성 완료"

# ── 완료 ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " 설치 완료!"
echo ""
echo " [터미널 1] LLM 서버 시작"
echo "   bash $PROJECT_DIR/run_server.sh"
echo ""
echo " [터미널 1] A100 GPU 서버 (FP16)"
echo "   bash $PROJECT_DIR/run_server_gpu.sh models/Qwen2.5-7B-Instruct"
echo ""
echo " [터미널 1] A100 GPU 서버 (4-bit, ~12GB VRAM)"
echo "   bash $PROJECT_DIR/run_server_gpu.sh models/Qwen2.5-7B-Instruct --load-in-4bit"
echo ""
echo " [터미널 2] Web UI 시작"
echo "   bash $PROJECT_DIR/run_ui.sh"
echo ""
echo " 브라우저: http://localhost:7860"
echo "======================================================"
