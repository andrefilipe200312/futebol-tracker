"""Um vídeo por fase de jogo, com todos os lances dessa fase seguidos (segmentos de fases.py).
Cada lance começa ANTES s antes e acaba DEPOIS s depois do segmento; lances que se sobrepõem juntam-se.
Barra no topo: equipa, fase (e tipo de bola parada), lance n/N e minuto do jogo. Ao lado de cada vídeo fica um
CSV com o minuto de cada lance e o segundo onde começa no vídeo, para encontrar o lance no jogo.
Uso: python cortes.py <video_jogo.mp4> <segmentos.csv> <saida_dir> [--equipa vermelho] [--antes 1.5] [--depois 1]"""
import csv, argparse
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

def fonte(tam=20):
    for p in ('/System/Library/Fonts/Supplemental/Arial.ttf', '/Library/Fonts/Arial.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        try: return ImageFont.truetype(p, tam)
        except OSError: pass
    return ImageFont.load_default()
FONTE = fonte(20)
def barra(txt):
    """Barra do topo (1280x34) com o texto, desenhada uma vez por segundo de vídeo (o OpenCV não escreve acentos)."""
    im = Image.new('RGB', (1280, 34), (25, 25, 25)); ImageDraw.Draw(im).text((12, 6), txt, font=FONTE, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

ap = argparse.ArgumentParser()
ap.add_argument('video'); ap.add_argument('segmentos'); ap.add_argument('saida')
ap.add_argument('--equipa', default='vermelho'); ap.add_argument('--antes', type=float, default=1.5); ap.add_argument('--depois', type=float, default=1.0)
a = ap.parse_args()
NOMES = {'organizacao_ofensiva': 'Organização ofensiva', 'organizacao_defensiva': 'Organização defensiva',
         'transicao_ofensiva': 'Transição ofensiva', 'transicao_defensiva': 'Transição defensiva',
         'bola_parada_ofensiva': 'Bola parada ofensiva', 'bola_parada_defensiva': 'Bola parada defensiva'}
out = Path(a.saida); out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
por_fase = defaultdict(list)
for r in csv.DictReader(open(a.segmentos)):
    por_fase[r['fase']].append((max(0, int(r['f_ini']) - int(a.antes * fps)), min(N - 1, int(r['f_fim']) + int(a.depois * fps)), r['tipo']))
def mmss(f): s = f / fps; return f'{int(s // 60):02d}:{int(s % 60):02d}'
for fase, L in por_fase.items():
    L.sort(); lances = []
    for i0, i1, tipo in L:                                    # junta lances sobrepostos
        if lances and i0 <= lances[-1][1]: lances[-1][1] = max(lances[-1][1], i1)
        else: lances.append([i0, i1, tipo])
    nome = f"{a.equipa} - {NOMES.get(fase, fase).lower()}"
    vw = cv2.VideoWriter(str(out / f'{nome}.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (1280, 720))
    idx = []; pos = 0; ultimo = None
    for n, (i0, i1, tipo) in enumerate(lances, 1):
        idx.append([n, mmss(i0), mmss(i1), round(pos / fps, 1), tipo]); cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
        for f in range(i0, i1 + 1):
            ok, im = cap.read()
            if not ok: break
            if im.shape[1] != 1280: im = cv2.resize(im, (1280, 720))
            txt = f"{a.equipa.upper()}   ·   {NOMES.get(fase, fase)}{' (' + tipo + ')' if tipo else ''}   ·   lance {n}/{len(lances)}   ·   {mmss(f)}"
            if txt != ultimo: cache = barra(txt); ultimo = txt
            im[:34] = cache
            vw.write(im); pos += 1
    vw.release()
    with open(out / f'{nome}.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['lance', 'inicio_jogo', 'fim_jogo', 'segundo_no_video', 'tipo']); w.writerows(idx)
    print(f'{nome}.mp4: {len(lances)} lances, {pos / fps:.0f} s', flush=True)
