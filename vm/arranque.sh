#!/bin/bash
# Script de arranque da VM (corre como root sempre que a VM liga).
# Lê a configuração dos metadados da VM, processa o jogo em N partes paralelas, envia os resultados para o
# bucket a cada 2 min e desliga a VM no fim. Se a VM for interrompida (Spot) e voltar a ligar, retoma:
# os trechos já enviados para o bucket são descarregados e saltados.
set -u
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
get() { curl -sf -H "Metadata-Flavor: Google" "$MD/$1" || echo "$2"; }
BUCKET=$(get bucket "")
VIDEO=$(get video "AFL-3Dv-S1-SnapYT.App.mp4")
FPS=$(get fps 5)
PARTES=$(get partes 4)
VIDEOS=$(get videos 1)
CH=300
W=/opt/jogo; mkdir -p $W/out/chunks; cd $W
LOG=$W/out/vm.log
exec >> $LOG 2>&1
echo "=== arranque $(date) bucket=$BUCKET video=$VIDEO fps=$FPS partes=$PARTES"

# 1) driver NVIDIA (a imagem Deep Learning da Google instala-o no primeiro arranque; pode demorar uns minutos)
for i in $(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,driver_version --format=csv || echo "AVISO: GPU não disponível, vai correr no CPU (lento)"

# 2) ambiente Python próprio (só da primeira vez)
if [ ! -f $W/.deps ]; then
  apt-get -qq update && apt-get -qq install -y python3-venv >/dev/null 2>&1
  python3 -m venv $W/venv
  $W/venv/bin/pip -q install --upgrade pip
  $W/venv/bin/pip -q install "ultralytics<8.4" scipy
  $W/venv/bin/pip -q uninstall -y opencv-python
  $W/venv/bin/pip -q install opencv-python-headless
  touch $W/.deps
fi
PY=$W/venv/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 3) código, vídeo e trechos já feitos (retoma)
gsutil -q cp gs://$BUCKET/pacote.zip . && rm -rf code && $PY -m zipfile -e pacote.zip code
[ -f video.mp4 ] || gsutil -q cp "gs://$BUCKET/$VIDEO" video.mp4
gsutil -q -m rsync -r gs://$BUCKET/resultado/chunks out/chunks 2>/dev/null || true
echo "trechos já feitos: $(ls out/chunks/*.csv 2>/dev/null | wc -l)"

# 4) envio periódico dos resultados (CSV, vídeos e registos) para o bucket
( while true; do sleep 120; gsutil -q -m rsync -r out gs://$BUCKET/resultado; done ) &
SYNC=$!

# 5) processamento: o jogo é dividido em PARTES blocos contíguos (múltiplos de 5 min), um processo por bloco
DUR=$($PY -c "import cv2;c=cv2.VideoCapture('video.mp4');print(int(c.get(7)/c.get(5))+1)")
NCH=$(( (DUR + CH - 1) / CH )); PER=$(( (NCH + PARTES - 1) / PARTES ))
EXTRA=""; [ "$VIDEOS" = "1" ] && EXTRA="--video"
PIDS=()
for p in $(seq 0 $((PARTES - 1))); do
  T0=$(( p * PER * CH )); T1=$(( (p + 1) * PER * CH ))
  [ $T0 -ge $DUR ] && continue
  echo "parte $p: $T0-$T1 s"
  $PY code/process.py video.mp4 out --fps $FPS --chunk $CH --t0 $T0 --t1 $T1 $EXTRA > out/parte_$p.log 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"

# 6) junta todos os trechos num só CSV, envia tudo e desliga
$PY - <<'EOF'
from pathlib import Path
parts = sorted(Path('out/chunks').glob('trecho_*.csv'))
with open('out/jogadores.csv', 'w') as fo:
    for i, p in enumerate(parts):
        lines = p.read_text().splitlines()
        fo.write('\n'.join(lines if i == 0 else lines[1:]) + '\n')
print('trechos juntos:', len(parts))
EOF
kill $SYNC 2>/dev/null
echo "=== fim $(date)"
date > out/FEITO
gsutil -q -m rsync -r out gs://$BUCKET/resultado
shutdown -h now
