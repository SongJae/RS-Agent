#!/usr/bin/env bash
# =============================================================================
# download_ml.sh  —  인터넷 연결 PC에서 실행
#
# ML 패키지 wheel 파일을 다운로드하여 packages-ml.tar.gz를 생성합니다.
# 생성된 파일을 USB에 담아 폐쇄망 PC에서 pack_conda_env.sh 실행 시 사용합니다.
#
# 사용법:
#   bash scripts/download_ml.sh
#
# 결과물:
#   packages-ml.tar.gz  (~2.8 GB)
#
# 폐쇄망 USB 구조:
#   /usb/
#   ├── RS-Agent/            ← 저장소 (git clone)
#   ├── packages-ml.tar.gz   ← 이 스크립트 생성물
#   └── rs-agent-env.tar.gz  ← pack_conda_env.sh 생성물 (있으면 install 바로 가능)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ML_WHEELS_DIR="$PROJECT_DIR/packages/ml-wheels"
OUTPUT="$PROJECT_DIR/packages-ml.tar.gz"

echo "======================================================"
echo " ML 패키지 다운로드"
echo " 저장 경로  : $ML_WHEELS_DIR"
echo " 출력 파일  : $OUTPUT"
echo "======================================================"

rm -rf "$ML_WHEELS_DIR"
mkdir -p "$ML_WHEELS_DIR"

echo ""
echo "[1/2] ML 패키지 wheel 다운로드 중 (Python 3.11 호환)..."
python3 -m pip download \
  torch \
  "transformers>=4.40.0" accelerate bitsandbytes \
  "sentence-transformers>=2.2.2" \
  "gradio>=6.0.0" \
  -d "$ML_WHEELS_DIR" \
  --python-version 3.11 \
  --only-binary=:all: \
  -q

echo "  → $(ls "$ML_WHEELS_DIR" | wc -l)개 파일 다운로드 완료"
du -sh "$ML_WHEELS_DIR"

echo ""
echo "[2/2] packages-ml.tar.gz 생성 중..."
tar -czf "$OUTPUT" -C "$PROJECT_DIR" packages/ml-wheels/

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "  → $OUTPUT ($SIZE)"

echo ""
echo "======================================================"
echo " USB에 복사할 파일:"
echo "   $OUTPUT  ($SIZE)"
echo ""
echo " 폐쇄망 PC에서 pack_conda_env.sh 실행 전:"
echo "   tar -xzf packages-ml.tar.gz -C RS-Agent/"
echo "   bash RS-Agent/scripts/pack_conda_env.sh"
echo "======================================================"
