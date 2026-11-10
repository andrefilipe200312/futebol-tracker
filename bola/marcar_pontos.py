"""Marcação manual de pontos do campo num frame do trecho (para a calibração imagem -> metros).
Pede, um de cada vez, pontos da área da baliza que se vê. O esquema no canto mostra qual é o ponto pedido.
"Esquerda/direita" = como se estivesses no meio-campo a olhar para essa baliza.

Teclas:  clique = marcar   s = saltar (não se vê)   u = desfazer   a/d = andar 10 frames (0,4 s)   z/x = andar 100 frames (4 s)
         Enter = terminar (precisa de >= 4 pontos)   q/Esc = sair sem gravar
Grava <saida>.json: {frame, pontos: [{nome, campo: [x, y] em metros a partir da linha de golo, imagem: [u, v]}]}
Coordenadas do campo aqui: x = distância à linha de golo (m, para dentro do campo), y = metros para a esquerda
do centro da baliza. A calibração converte para o referencial do campo inteiro.
"campo_m" = os mesmos pontos no referencial do campo inteiro (origem no centro, x para a baliza da DIREITA de quem
está na bancada da câmara, y para a lateral do fundo), com comprimento 100 m.
Uso: python marcar_pontos.py <trecho.mp4> <saida.json> <frame_inicial> <esquerda|direita>
  esquerda/direita = de que lado está esta baliza, visto da bancada onde está a câmara"""
import sys, json
import cv2, numpy as np

video, out = sys.argv[1], sys.argv[2]
f = int(sys.argv[3]) if len(sys.argv) > 3 else 0
BALIZA = sys.argv[4] if len(sys.argv) > 4 else 'direita'; assert BALIZA in ('esquerda', 'direita')
def campo_m(x, y): return [50 - x, y] if BALIZA == 'direita' else [-50 + x, -y]
PONTOS = [  # nome, (x a partir da linha de golo, y para a esquerda)
    ('centro da baliza (meio da linha de golo)', (0, 0)),
    ('pequena área: canto ESQUERDO na linha de golo', (0, 9.16)),
    ('pequena área: canto DIREITO na linha de golo', (0, -9.16)),
    ('pequena área: canto da frente ESQUERDO', (5.5, 9.16)),
    ('pequena área: canto da frente DIREITO', (5.5, -9.16)),
    ('marca de penálti', (11, 0)),
    ('grande área: canto ESQUERDO na linha de golo', (0, 20.16)),
    ('grande área: canto DIREITO na linha de golo', (0, -20.16)),
    ('grande área: canto da frente ESQUERDO', (16.5, 20.16)),
    ('grande área: canto da frente DIREITO', (16.5, -20.16)),
    ('arco da área: ponto mais afastado da baliza', (20.15, 0)),
]
cap = cv2.VideoCapture(video); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
def ler(f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); return im

def esquema(i):
    """Desenho da área (vista de cima, baliza em cima) com o ponto pedido a vermelho."""
    S, W, H = 6, 300, 170; e = np.full((H, W, 3), (40, 110, 40), np.uint8)
    P = lambda x, y: (int(W / 2 - y * S), int(12 + x * S))          # esquerda do jogador = esquerda do desenho
    cv2.line(e, P(0, 30), P(0, -30), (255, 255, 255), 1)
    cv2.rectangle(e, P(0, 3.66), P(-2, -3.66), (255, 255, 255), 1)                    # baliza
    cv2.rectangle(e, P(0, 9.16), P(5.5, -9.16), (255, 255, 255), 1)
    cv2.rectangle(e, P(0, 20.16), P(16.5, -20.16), (255, 255, 255), 1)
    a = np.degrees(np.arccos(5.5 / 9.15)); c = P(11, 0)
    cv2.ellipse(e, c, (int(9.15 * S), int(9.15 * S)), 90, -a, a, (255, 255, 255), 1)
    cv2.circle(e, c, 2, (255, 255, 255), -1)
    for j, (_, (x, y)) in enumerate(PONTOS):
        cv2.circle(e, P(x, y), 6 if j == i else 3, (0, 0, 255) if j == i else (200, 200, 200), -1 if j == i else 1)
    cv2.putText(e, 'baliza', (W // 2 - 22, 9), 0, .35, (255, 255, 255), 1)
    return e

marc = []; i = 0; rato = (0, 0); im = ler(f)
def clique(ev, x, y, *_):
    global i, rato
    rato = (x, y)
    if ev == cv2.EVENT_LBUTTONDOWN and i < len(PONTOS):
        marc.append((i, (x, y))); i += 1
cv2.namedWindow('marcar', cv2.WINDOW_AUTOSIZE); cv2.setMouseCallback('marcar', clique)
while True:
    v = im.copy()
    for j, (u, w) in marc:
        cv2.drawMarker(v, (u, w), (0, 0, 255), cv2.MARKER_CROSS, 14, 2); cv2.putText(v, str(j + 1), (u + 6, w - 6), 0, .5, (0, 0, 255), 2)
    x, y = rato; r = 20                                                   # lupa 5x junto ao cursor
    p = cv2.copyMakeBorder(im, r, r, r, r, cv2.BORDER_CONSTANT)[y:y + 2 * r, x:x + 2 * r]
    if p.shape[:2] == (2 * r, 2 * r):
        z = cv2.resize(p, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.line(z, (100, 85), (100, 115), (0, 0, 255), 1); cv2.line(z, (85, 100), (115, 100), (0, 0, 255), 1)
        zx = x + 30 if x + 230 < 1280 else x - 230; zy = min(max(y - 100, 0), 720 - 200); v[zy:zy + 200, zx:zx + 200] = z
    e = esquema(i); v[40:40 + e.shape[0], 1280 - e.shape[1] - 5:1275] = e
    cv2.rectangle(v, (0, 0), (1280, 34), (25, 25, 25), -1)
    txt = (f'{i + 1}/{len(PONTOS)}: {PONTOS[i][0]}' if i < len(PONTOS) else 'todos pedidos: Enter para gravar') + \
          f'  baliza da {BALIZA}   [s saltar  u desfazer  a/d z/x frame  Enter gravar]  frame {f} ({f / fps:.0f}s)  marcados {len(marc)}'
    cv2.putText(v, txt, (8, 22), 0, .5, (255, 255, 255), 1)
    cv2.imshow('marcar', v); k = cv2.waitKey(30) & 0xFF
    if k in (ord('q'), 27): print('saiu sem gravar'); break
    if k == ord('s') and i < len(PONTOS): i += 1
    if k == ord('u'):
        if marc and marc[-1][0] == i - 1: marc.pop()
        i = max(0, i - 1)
    if k in (ord('a'), ord('d'), ord('z'), ord('x')):          # passos múltiplos de 10 (frames analisados)
        f = int(np.clip(f + {ord('a'): -10, ord('d'): 10, ord('z'): -100, ord('x'): 100}[k], 0, N - 1)); f -= f % 10
        im = ler(f); marc = []; i = 0
    if k in (13, 10):
        if len(marc) < 4: print('precisa de pelo menos 4 pontos'); continue
        json.dump(dict(frame=f, baliza=BALIZA, pontos=[dict(nome=PONTOS[j][0], campo=list(PONTOS[j][1]), campo_m=campo_m(*PONTOS[j][1]), imagem=list(p)) for j, p in marc]),
                  open(out, 'w'), indent=1, ensure_ascii=False)
        print(f'gravado: {out} ({len(marc)} pontos no frame {f})'); break
cv2.destroyAllWindows()
