#!/usr/bin/env bash
# =============================================================================
# install_offline.sh  —  폐쇄망 PC에서 실행
#
# rs-agent-env.tar.gz를 압축 해제하고 RS-Agent를 바로 실행할 수 있도록 설정합니다.
# 인터넷 연결이 전혀 없어도 됩니다.
#
# 사용법:
#   # rs-agent-env.tar.gz와 RS-Agent/ 가 같은 디렉토리에 있는 경우
#   bash RS-Agent/scripts/install_offline.sh
#
#   # tar.gz 경로를 직접 지정
#   bash RS-Agent/scripts/install_offline.sh /media/usb/rs-agent-env.tar.gz
# =============================================================================
set -euo pipefail

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_TAR="${1:-$(dirname "$PROJECT_DIR")/rs-agent-env.tar.gz}"
INSTALL_DIR="$HOME/rs-agent-env"

echo "======================================================"
echo " RS-Agent 오프라인 설치"
echo " 환경 압축 파일 : $ENV_TAR"
echo " 설치 경로      : $INSTALL_DIR"
echo " 프로젝트 경로  : $PROJECT_DIR"
echo "======================================================"

# ── 1. tar.gz 확인 ───────────────────────────────────────────────────────────
if [ ! -f "$ENV_TAR" ]; then
  echo "[ERROR] 환경 파일을 찾을 수 없습니다: $ENV_TAR"
  echo ""
  echo "  다음 중 하나를 확인하세요:"
  echo "    1) rs-agent-env.tar.gz를 RS-Agent/ 상위 디렉토리에 복사"
  echo "    2) bash install_offline.sh /경로/rs-agent-env.tar.gz"
  exit 1
fi

# ── 2. 기존 환경 제거 후 압축 해제 ───────────────────────────────────────────
echo ""
echo "[1/4] conda 환경 압축 해제 중..."
if [ -d "$INSTALL_DIR" ]; then
  echo "  기존 환경 디렉토리 삭제 중: $INSTALL_DIR"
  rm -rf "$INSTALL_DIR"
fi
mkdir -p "$INSTALL_DIR"
tar -xzf "$ENV_TAR" -C "$INSTALL_DIR"
echo "  → 압축 해제 완료"

# ── 3. 환경 활성화 준비 (conda-unpack) ───────────────────────────────────────
echo ""
echo "[2/4] 환경 경로 재설정 중 (conda-unpack)..."
source "$INSTALL_DIR/bin/activate"
conda-unpack
echo "  → 재설정 완료"

# ── 4. 환경변수 설정 ─────────────────────────────────────────────────────────
echo ""
echo "[3/4] 오프라인 환경변수 설정..."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── 5. 실행 스크립트 생성 ─────────────────────────────────────────────────────
echo ""
echo "[4/4] 실행 스크립트 생성 중..."

cat > "$PROJECT_DIR/run_server.sh" <<EOF
#!/usr/bin/env bash
# LLM 서버 실행 (Mock 백엔드)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source "$INSTALL_DIR/bin/activate"
cd "$PROJECT_DIR"
python scripts/start_local_server.py --backend mock --port 11434
EOF

cat > "$PROJECT_DIR/run_server_gpu.sh" <<EOF
#!/usr/bin/env bash
# LLM 서버 실행 (A100 GPU 백엔드 — 실제 LLM 추론)
# 사용법: bash run_server_gpu.sh /models/Qwen2.5-7B-Instruct [--load-in-4bit]
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source "$INSTALL_DIR/bin/activate"
cd "$PROJECT_DIR"
python scripts/start_local_server.py --backend transformers --model-path "\${1:?모델 경로 필요}" \${2:-} --port 11434
EOF

cat > "$PROJECT_DIR/run_ui.sh" <<EOF
#!/usr/bin/env bash
# Gradio UI 실행
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source "$INSTALL_DIR/bin/activate"
cd "$PROJECT_DIR"
python app.py --port 7860
EOF

chmod +x "$PROJECT_DIR/run_server.sh" \
         "$PROJECT_DIR/run_server_gpu.sh" \
         "$PROJECT_DIR/run_ui.sh"

echo "  → run_server.sh      생성 완료"
echo "  → run_server_gpu.sh  생성 완료"
echo "  → run_ui.sh          생성 완료"

# ── 완료 메시지 ──────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " 설치 완료! 실행 방법:"
echo ""
echo " [터미널 1] LLM 서버 시작 (Mock 모드, GPU 불필요)"
echo "   bash $PROJECT_DIR/run_server.sh"
echo ""
echo " [터미널 1] LLM 서버 시작 (A100 GPU, FP16)"
echo "   bash $PROJECT_DIR/run_server_gpu.sh /models/Qwen2.5-7B-Instruct"
echo ""
echo " [터미널 1] LLM 서버 시작 (A100 GPU, 4-bit 양자화)"
echo "   bash $PROJECT_DIR/run_server_gpu.sh /models/Qwen2.5-7B-Instruct --load-in-4bit"
echo ""
echo " [터미널 2] Web UI 시작"
echo "   bash $PROJECT_DIR/run_ui.sh"
echo ""
echo " 브라우저: http://localhost:7860"
echo "======================================================"
