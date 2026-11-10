"""Modelo físico de câmara para o plano do campo (Z=0).
Coordenadas do campo em metros: origem no centro, x ao longo do comprimento (para a baliza da direita
na imagem inicial), y na largura (positivo = lateral do fundo), z para cima.
Parâmetros: posição (cx, cy, cz), rumo, inclinação, rotação (rad), focal (px). Distorção radial k1 fixa."""
import numpy as np

W_IMG, H_IMG = 1280, 720
PP = np.array([W_IMG / 2, H_IMG / 2]); FD = 1000.0; K1 = 0.2   # distorção (estimada pelas linhas retas)

def rot(yaw, tilt, roll):
    # câmara: z_cam para a frente, x_cam para a direita, y_cam para baixo
    cy, sy = np.cos(yaw), np.sin(yaw); ct, st = np.cos(tilt), np.sin(tilt); cr, sr = np.cos(roll), np.sin(roll)
    f = np.array([cy * ct, sy * ct, -st])                 # direção de visão (tilt>0 = para baixo)
    r = np.array([sy, -cy, 0.0])                          # direita
    d = np.cross(f, r)                                     # baixo
    R0 = np.vstack([r, d, f])
    Rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rr @ R0

def homography(p):
    """p = [cx, cy, cz, yaw, tilt, roll, focal] -> H (campo xy -> imagem não distorcida)."""
    C = np.array(p[:3]); R = rot(*p[3:6]); f = p[6]
    K = np.array([[f, 0, PP[0]], [0, f, PP[1]], [0, 0, 1]])
    return K @ np.c_[R[:, 0], R[:, 1], -R @ C]

def undistort(pts):
    pts = np.asarray(pts, float); x = (pts[:, 0] - PP[0]) / FD; y = (pts[:, 1] - PP[1]) / FD; s = 1 + K1 * (x * x + y * y)
    return np.c_[x * s * FD + PP[0], y * s * FD + PP[1]]

def distort(pts):
    pts = np.asarray(pts, float); q = pts.copy()
    for _ in range(15):
        x = (q[:, 0] - PP[0]) / FD; y = (q[:, 1] - PP[1]) / FD; s = 1 + K1 * (x * x + y * y)
        q = np.c_[(pts[:, 0] - PP[0]) / s + PP[0], (pts[:, 1] - PP[1]) / s + PP[1]]
    return q

def look(C, target, roll=0.0):
    d = np.asarray(target, float) - C; yaw = np.arctan2(d[1], d[0]); tilt = np.arctan2(-d[2], np.hypot(d[0], d[1]))
    return yaw, tilt, roll
