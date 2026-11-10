"""Passo 1 do tracking da bola: deteção em todos os frames (lento; corre uma vez).
Para cada frame guarda, numa linha de <saida>.jsonl:
  balls   candidatos a bola (classe COCO 32) com conf >= --conf
  persons pessoas com conf >= 0.25 (só para tapar no cálculo do movimento da câmara)
  G       homografia frame anterior -> este frame (movimento da câmara), ou null se falhar
Retoma onde parou. O passo 2 (seguir_bola.py) lê este ficheiro.
Uso: python detetar_bola.py <video> <saida.jsonl> [--passo 1] [--model yolo26s_1280_openvino_model] [--imgsz 736,1280] [--conf 0.03] [--t0 0] [--t1 fim]
--model: nome em pesos/ (.pt ou pasta *_openvino_model, exportada com o tamanho fixo de --imgsz). Escolhido em comparar_modelos.py / avaliar.py."""
import sys, json, time, argparse
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO, RTDETR

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('video'); ap.add_argument('out')
ap.add_argument('--model', default='yolo26s_1280_openvino_model'); ap.add_argument('--imgsz', default='736,1280', help='lado maior, ou altura,largura')
ap.add_argument('--conf', type=float, default=0.03)
ap.add_argument('--passo', type=int, default=1, help='analisa 1 frame em cada PASSO (o movimento da câmara é entre frames analisados)')
ap.add_argument('--t0', type=float, default=0); ap.add_argument('--t1', type=float, default=None)
a = ap.parse_args()

def camera_motion(prev_gray, gray, prev_boxes):
    """Homografia prev -> atual (igual a tracker.camera_motion, sem a dependência de scipy). None se falhar."""
    if prev_gray is None: return None
    m = np.full(prev_gray.shape, 255, np.uint8); m[:int(m.shape[0] * .04)] = 0
    for x1, y1, x2, y2 in prev_boxes: m[max(0, int(y1) - 4):int(y2) + 4, max(0, int(x1) - 4):int(x2) + 4] = 0
    p = cv2.goodFeaturesToTrack(prev_gray, 800, 0.01, 12, mask=m)
    if p is None or len(p) < 30: return None
    q, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p, None, winSize=(21, 21), maxLevel=4)
    ok = st.ravel() == 1
    if ok.sum() < 30: return None
    G, inl = cv2.findHomography(p[ok], q[ok], cv2.RANSAC, 2.0)
    if G is None or inl.sum() < 25: return None
    return G

imgsz = [int(v) for v in a.imgsz.split(',')]; imgsz = imgsz[0] if len(imgsz) == 1 else imgsz
w = HERE / 'pesos' / a.model
if not w.exists(): w = w.with_suffix('.pt')
model = RTDETR(str(w)) if 'rtdetr' in a.model else YOLO(str(w), task='detect')
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
f0 = int(round(a.t0 * fps)); f1 = min(N, int(round(a.t1 * fps)) if a.t1 else N)
out = Path(a.out)
lines = [l for l in open(out).read().splitlines() if l.strip()] if out.exists() else []
start = json.loads(lines[-1])['f'] if lines else f0   # relê o último frame já feito (só para o movimento da câmara)
fo = open(out, 'a')
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
prev_gray = prev_boxes = None; t = time.time(); n = 0
print(f'{a.model} @ {a.imgsz}: frames {start}-{f1} de {N} ({fps:.0f} fps)', flush=True)
for f in range(start, f1):
    if (f - f0) % a.passo:
        if not cap.grab(): break
        continue
    ok, im = cap.read()
    if not ok: break
    r = model.predict(im, imgsz=imgsz, classes=[0, 32], conf=a.conf, verbose=False)[0]
    B = r.boxes.xyxy.tolist(); C = r.boxes.conf.tolist(); K = r.boxes.cls.tolist()
    balls = [[round(v, 1) for v in b] + [round(c, 3)] for b, c, k in zip(B, C, K) if k == 32]
    persons = [[round(v, 1) for v in b] for b, c, k in zip(B, C, K) if k == 0 and c >= 0.25]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    G = camera_motion(prev_gray, gray, prev_boxes or [])
    if f > start or not lines:
        fo.write(json.dumps(dict(f=f, t=round(f / fps, 3), balls=balls, persons=persons,
                                 G=None if G is None else [round(v, 6) for v in G.ravel()])) + '\n'); fo.flush()
    prev_gray, prev_boxes = gray, persons; n += 1
    if n % 100 == 0: print(f'  frame {f}/{f1}  {(time.time() - t) / n:.2f}s/frame', flush=True)
print('FEITO', flush=True)
