Miniconda3 설치 파일 (분할 저장)
====================================
원본 파일(155 MB)이 git 100MB 한도를 초과하므로 2개로 분할 저장됩니다.

  Miniconda3-latest-Linux-x86_64.sh.partaa  (90 MB)
  Miniconda3-latest-Linux-x86_64.sh.partab  (65 MB)

복원 방법:
  cat Miniconda3-latest-Linux-x86_64.sh.part* > Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh

이 작업은 pack_conda_env.sh 와 install_offline.sh 에서 자동으로 수행됩니다.
