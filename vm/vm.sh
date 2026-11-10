#!/bin/bash
# Controlo da VM de processamento. Correr na Cloud Shell da Google Cloud, a partir do repositório clonado:
#   git clone <url-do-repositório> futebol && cd futebol/vm
# O código enviado para a VM é o do repositório (versão atual, com git pull antes se houver alterações):
# 'lancar' e 'retomar' empacotam-no (pacote.zip) e enviam-no para o bucket. O vídeo vai à parte para o bucket.
#
#   PROJECT=<id-do-projeto> bash vm.sh preparar   # ativa APIs, cria o bucket, verifica a quota de GPU
#   (carrega o vídeo para o bucket — ver instruções)
#   PROJECT=<id-do-projeto> bash vm.sh lancar     # cria a VM Spot com T4; começa a processar sozinha
#   PROJECT=<id-do-projeto> bash vm.sh estado     # progresso (trechos feitos, últimas linhas do registo)
#   PROJECT=<id-do-projeto> bash vm.sh retomar    # volta a ligar a VM se foi interrompida (Spot)
#   PROJECT=<id-do-projeto> bash vm.sh descarregar # copia os resultados para a Cloud Shell (./resultado)
#   PROJECT=<id-do-projeto> bash vm.sh apagar     # apaga a VM (o bucket com os resultados fica)
#
# Opções (variáveis de ambiente): ZONE (europe-west1-b), BUCKET (<projeto>-futebol), VM (futebol-vm),
# VIDEO (AFL-3Dv-S1-SnapYT.App.mp4), FPS (5), PARTES (4), VIDEOS (1 = grava vídeos por trecho),
# MAQUINA (n1-standard-8), SPOT (1 = modo Spot, mais barato e interrompível; 0 = normal)
set -eu
PROJECT=${PROJECT:?"define PROJECT=<id do projeto>"}
ZONE=${ZONE:-europe-west1-b}; REGION=${ZONE%-*}
BUCKET=${BUCKET:-$PROJECT-futebol}; VM=${VM:-futebol-vm}
VIDEO=${VIDEO:-AFL-3Dv-S1-SnapYT.App.mp4}; FPS=${FPS:-5}; PARTES=${PARTES:-4}; VIDEOS=${VIDEOS:-1}
MAQUINA=${MAQUINA:-n1-standard-8}; SPOT=${SPOT:-1}
gcloud config set project "$PROJECT" -q >/dev/null
REPO=$(cd "$(dirname "$0")/.." && git rev-parse --show-toplevel)

enviar_codigo() {
  # empacota a versão atual do repositório (HEAD) e envia para o bucket; a VM descarrega-a em cada arranque
  (cd "$REPO" && git archive --format=zip -o /tmp/pacote.zip HEAD)
  gcloud storage cp /tmp/pacote.zip gs://$BUCKET/pacote.zip
  echo "Código enviado: $(cd "$REPO" && git log -1 --format='%h %s')"
}

case "${1:-}" in
preparar)
  gcloud services enable compute.googleapis.com storage.googleapis.com
  gcloud storage buckets describe gs://$BUCKET >/dev/null 2>&1 || gcloud storage buckets create gs://$BUCKET --location=$REGION
  echo "Bucket: gs://$BUCKET"
  echo "Quota de GPU T4 na região $REGION e global (precisa de >= 1 em ambas):"
  gcloud compute regions describe $REGION --format=json | python3 -c "import json,sys; q=[x for x in json.load(sys.stdin)['quotas'] if x['metric'] in ('NVIDIA_T4_GPUS','PREEMPTIBLE_NVIDIA_T4_GPUS')]; [print(' ',x['metric'],'limite',x['limit'],'em uso',x['usage']) for x in q]"
  gcloud compute project-info describe --format=json | python3 -c "import json,sys; q=[x for x in json.load(sys.stdin)['quotas'] if x['metric']=='GPUS_ALL_REGIONS']; [print(' ',x['metric'],'limite',x['limit'],'em uso',x['usage']) for x in q] or print('  GPUS_ALL_REGIONS: não listada')"
  echo
  echo "Ficheiros no bucket (precisa de $VIDEO):"; gcloud storage ls gs://$BUCKET/ || true
  ;;
lancar)
  gcloud storage ls "gs://$BUCKET/$VIDEO" >/dev/null 2>&1 || { echo "Falta gs://$BUCKET/$VIDEO — carrega o vídeo primeiro."; exit 1; }
  enviar_codigo
  IMG=$(gcloud compute images list --project=deeplearning-platform-release --no-standard-images \
        --filter="family~^common-cu AND -family~arm" --sort-by=~creationTimestamp --format="value(family)" | head -1)
  echo "Imagem (com driver NVIDIA): $IMG"
  SPOTOPT=""; [ "$SPOT" = "1" ] && SPOTOPT="--provisioning-model=SPOT --instance-termination-action=STOP"
  gcloud compute instances create $VM --zone=$ZONE --machine-type=$MAQUINA \
    --accelerator=type=nvidia-tesla-t4,count=1 --maintenance-policy=TERMINATE $SPOTOPT \
    --image-family=$IMG --image-project=deeplearning-platform-release --boot-disk-size=100GB \
    --scopes=cloud-platform \
    --metadata=install-nvidia-driver=True,bucket=$BUCKET,video=$VIDEO,fps=$FPS,partes=$PARTES,videos=$VIDEOS \
    --metadata-from-file=startup-script="$REPO/vm/arranque.sh"
  echo "VM criada. Vai instalar tudo e começar (~10 min até aos primeiros trechos). Acompanha com: bash vm.sh estado"
  ;;
estado)
  gcloud compute instances describe $VM --zone=$ZONE --format="value(status)" 2>/dev/null | sed 's/^/VM: /' || echo "VM: não existe"
  echo "Trechos feitos (de 5 min): $(gcloud storage ls "gs://$BUCKET/resultado/chunks/*.csv" 2>/dev/null | wc -l)"
  gcloud storage ls gs://$BUCKET/resultado/FEITO >/dev/null 2>&1 && echo "ACABOU: resultados em gs://$BUCKET/resultado/"
  for f in vm.log parte_0.log parte_1.log parte_2.log parte_3.log; do
    gcloud storage cat gs://$BUCKET/resultado/$f 2>/dev/null | grep -v -i warn | tail -2 | sed "s/^/[$f] /" || true
  done
  ;;
retomar)
  enviar_codigo
  gcloud compute instances add-metadata $VM --zone=$ZONE --metadata-from-file=startup-script="$REPO/vm/arranque.sh"
  gcloud compute instances start $VM --zone=$ZONE ;;
descarregar)
  mkdir -p resultado && gcloud storage cp -r gs://$BUCKET/resultado/* resultado/ && du -sh resultado ;;
apagar)
  gcloud compute instances delete $VM --zone=$ZONE ;;
*)
  sed -n '2,24p' "$0" ;;
esac
