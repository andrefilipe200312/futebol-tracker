"""Processamento com visualização ao vivo: lê o vídeo, deteta, classifica a equipa e mostra cada frame
numa janela assim que fica pronto (também grava o resultado em mp4). Tecla q/Esc para parar.
Uso: python live.py <video> <teams_model.npz> <t_ini> <t_fim> [fps=5] [saida.mp4] [mapa grupo=rotulo ...]"""
import sys, cv2, numpy as np
from ultralytics import YOLO
from pathlib import Path
from sig import torso_sig
from teams import label_frame, draw

video, model_npz, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
SFPS = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
out = sys.argv[6] if len(sys.argv) > 6 else 'live.mp4'
lab_of = dict(a.split('=') for a in sys.argv[7:]) or {'0': 'rosa', '1': 'rosa', '4': 'vermelho', '5': 'arbitro', '2': 'fora', '3': 'fora'}
C = np.load(model_npz)['C']
model = YOLO(str(Path(__file__).parent / 'ov1280_openvino_model'), task='detect')
cap = cv2.VideoCapture(video); vfps = cap.get(cv2.CAP_PROP_FPS); step = vfps / SFPS
cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000); pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), SFPS, (1280, 720))
cv2.namedWindow('equipas', cv2.WINDOW_NORMAL); cv2.resizeWindow('equipas', 1280, 720)
k = 0
while True:
    target = int(round(t0 * vfps + k * step)); k += 1
    if target / vfps >= t1: break
    while pos < target: cap.grab(); pos += 1
    ok, im = cap.read(); pos += 1
    if not ok: break
    r = model.predict(im, imgsz=[736, 1280], conf=0.25, classes=[0], verbose=False)[0]
    lab = label_frame(im, [(b, torso_sig(im, b)[0]) for b in r.boxes.xyxy.tolist()], C, lab_of)
    frame = draw(im, lab, target / vfps); vw.write(frame)
    cv2.imshow('equipas', frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
vw.release(); cv2.destroyAllWindows(); print('gravado em', out)
