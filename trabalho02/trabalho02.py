
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mp_py
from mediapipe.tasks.python import vision as _mp_vision
import pygame
import time
import sys
import os
from collections import deque

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk   # pip install Pillow

# ──────────────────────────────────────────────
# Calibração da câmera
# ──────────────────────────────────────────────
CALIB_PATH = r"calibracao\calibracao_webcam_casa.npz"
try:
    _c = np.load(CALIB_PATH)
    CAM_MTX  = _c["mtx"].astype(np.float64)
    CAM_DIST = _c["dist"].astype(np.float64)
    CALIB_STATUS = "✔ Calibração carregada"
except Exception:
    CAM_MTX  = np.array([[650,0,320],[0,650,240],[0,0,1]], dtype=np.float64)
    CAM_DIST = np.zeros((1,5), dtype=np.float64)
    CALIB_STATUS = "⚠ Valores padrão (sem calibração)"

# ──────────────────────────────────────────────
# ArUco
# ──────────────────────────────────────────────
ARUCO_DICT     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
ARUCO_PARAMS   = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
MARKER_SIZE_M  = 0.06

# ──────────────────────────────────────────────
# Áudio — Ocarina
# ──────────────────────────────────────────────
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
SAMPLE_RATE = 44100

NOTE_FREQS  = {1:261.63,2:293.66,3:329.63,4:349.23,5:392.00,6:440.00,7:493.88}
NOTE_NAMES  = {1:"Dó",2:"Ré",3:"Mi",4:"Fá",5:"Sol",6:"Lá",7:"Si"}
NOTE_COLORS_CV = {
    1:(255,80,80),2:(255,160,60),3:(255,220,0),
    4:(80,200,80),5:(60,180,255),6:(140,80,255),7:(255,80,200),
}
NOTE_COLORS_HEX = {
    1:"#ff5050",2:"#ffa03c",3:"#ffdc00",
    4:"#50c850",5:"#3cb4ff",6:"#8c50ff",7:"#ff50c8",
}

def _make_sound(freq, duration=0.4):
    t = np.linspace(0, duration, int(SAMPLE_RATE*duration), endpoint=False)
    wave = np.sin(2*np.pi*freq*t) * np.linspace(1.0,0.0,int(SAMPLE_RATE*duration))
    mono = (wave*32767).astype(np.int16)
    stereo = np.column_stack([mono, mono])
    try:
        return pygame.sndarray.make_sound(stereo)
    except ValueError:
        return pygame.sndarray.make_sound(mono)

_sound_cache = {nid: _make_sound(freq) for nid, freq in NOTE_FREQS.items()}

# ──────────────────────────────────────────────
# MediaPipe Hands
# ──────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_mp_base_opts = _mp_py.BaseOptions(model_asset_path=_MODEL_PATH)
_mp_hand_opts = _mp_vision.HandLandmarkerOptions(
    base_options=_mp_base_opts,
    running_mode=_mp_vision.RunningMode.VIDEO,
    num_hands=3,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
_hands_model  = _mp_vision.HandLandmarker.create_from_options(_mp_hand_opts)
_mp_frame_ts  = 0

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

# ──────────────────────────────────────────────
# Helpers de desenho (OpenCV)
# ──────────────────────────────────────────────
FONT    = cv2.FONT_HERSHEY_SIMPLEX
C_W     = (255,255,255); C_BLK=(0,0,0)
C_YEL   = (0,220,255);   C_GRN=(0,220,80)
C_CYN   = (255,200,0)

def put_text(img, text, pos, scale=0.6, color=C_W, thickness=1, bg=True):
    (tw,th),bl = cv2.getTextSize(text, FONT, scale, thickness)
    x,y = pos
    if bg:
        cv2.rectangle(img,(x-2,y-th-bl-2),(x+tw+2,y+2),C_BLK,-1)
    cv2.putText(img,text,(x,y),FONT,scale,color,thickness,cv2.LINE_AA)

def draw_axes(img, rvec, tvec, length=0.03):
    pts,_ = cv2.projectPoints(
        np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,-length]]),
        rvec, tvec, CAM_MTX, CAM_DIST)
    pts = pts.reshape(-1,2).astype(int)
    cv2.arrowedLine(img,tuple(pts[0]),tuple(pts[1]),(0,0,255),2,tipLength=0.3)
    cv2.arrowedLine(img,tuple(pts[0]),tuple(pts[2]),(0,255,0),2,tipLength=0.3)
    cv2.arrowedLine(img,tuple(pts[0]),tuple(pts[3]),(255,0,0),2,tipLength=0.3)

def _marker_center(corners):
    return corners[0].reshape(-1,2).mean(axis=0)

def _pose_from_corners(corners):
    obj_pts = np.array([
        [-MARKER_SIZE_M/2, MARKER_SIZE_M/2,0],
        [ MARKER_SIZE_M/2, MARKER_SIZE_M/2,0],
        [ MARKER_SIZE_M/2,-MARKER_SIZE_M/2,0],
        [-MARKER_SIZE_M/2,-MARKER_SIZE_M/2,0],
    ], dtype=np.float64)
    img_pts = corners[0].astype(np.float64)
    ok,rvec,tvec = cv2.solvePnP(obj_pts,img_pts,CAM_MTX,CAM_DIST,
                                 flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec,tvec) if ok else (None,None)

# ══════════════════════════════════════════════
# MODO 1 — METROLOGIA
# ══════════════════════════════════════════════
_metro_history = deque(maxlen=15)

def process_metrology(frame, app):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    out = frame.copy()
    info_lines = []

    if ids is not None and len(ids) >= 2:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        tvecs = []
        for c, mid in zip(corners, ids.flatten()):
            rvec,tvec = _pose_from_corners([c])
            if tvec is not None:
                draw_axes(out, rvec, tvec)
                tvecs.append(tvec.flatten())
                ctr = _marker_center([c]).astype(int)
                put_text(out, f"ID:{mid}", (int(ctr[0]),int(ctr[1])+30),
                         scale=0.55, color=C_CYN)
        if len(tvecs) >= 2:
            dist_m = float(np.linalg.norm(tvecs[0]-tvecs[1]))
            _metro_history.append(dist_m)
            smooth = float(np.mean(_metro_history))
            info_lines.append(f"Distância: {smooth*100:.1f} cm  ({smooth*1000:.0f} mm)")
            c0 = _marker_center([corners[0]]).astype(int)
            c1 = _marker_center([corners[1]]).astype(int)
            cv2.line(out, tuple(c0), tuple(c1), C_GRN, 2)
            mid_px = ((c0+c1)/2).astype(int)
            put_text(out, f"{smooth*100:.1f} cm",
                     (int(mid_px[0]),int(mid_px[1])-15), scale=0.75, color=C_GRN)
            app.set_status(f"📏 {smooth*100:.1f} cm | {smooth*1000:.0f} mm")
    elif ids is not None and len(ids)==1:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        rvec,tvec = _pose_from_corners([corners[0]])
        if tvec is not None:
            draw_axes(out, rvec, tvec)
            z = float(tvec.flatten()[2])
            info_lines.append(f"Distância ao marcador: {z*100:.1f} cm")
            app.set_status(f"1 marcador — {z*100:.1f} cm")
        info_lines.append("Mostre 2 marcadores para medir distância")
    else:
        _metro_history.clear()
        info_lines.append("Nenhum marcador detectado")
        app.set_status("Aguardando marcadores ArUco…")

    for i, line in enumerate(info_lines):
        put_text(out, line, (10, 30+i*28), scale=0.6, color=C_YEL)
    return out

# ══════════════════════════════════════════════
# MODO 2 — OCARINA VIRTUAL
# ══════════════════════════════════════════════
OCARINA_BODY_ID = 0
OCARINA_HOLE_IDS = [1,2,3,4,5,6,7]
_note_playing: dict = {}
_NOTE_COOLDOWN = 0.5
_hole_absent: dict = {i:0 for i in OCARINA_HOLE_IDS}
_ABSENT_THRESHOLD = 5

def _draw_ocarina_body(out, body_corners, body_rvec, body_tvec, hole_states):
    W,H,D = 0.14,0.06,0.0
    pts3d = np.float32([[-W/2,-H/2,D],[W/2,-H/2,D],[W/2,H/2,D],[-W/2,H/2,D]])
    pts2d,_ = cv2.projectPoints(pts3d, body_rvec, body_tvec, CAM_MTX, CAM_DIST)
    pts2d = pts2d.reshape(-1,2).astype(int)
    cv2.fillConvexPoly(out, pts2d, (50,140,200))
    cv2.polylines(out, [pts2d], True, (200,220,255), 2)
    xs = np.linspace(-0.40*W, 0.40*W, 7)
    for i, hid in enumerate(OCARINA_HOLE_IDS):
        hpt,_ = cv2.projectPoints(np.float32([[xs[i],0,D]]),
                                   body_rvec, body_tvec, CAM_MTX, CAM_DIST)
        hp = tuple(hpt.reshape(2).astype(int))
        covered = hole_states.get(hid, False)
        fill = NOTE_COLORS_CV[hid] if covered else (20,20,20)
        cv2.circle(out, hp, 8, fill, -1)
        cv2.circle(out, hp, 8, (220,220,220), 1)
        if covered:
            put_text(out, NOTE_NAMES[hid], (hp[0]-10,hp[1]-16),
                     scale=0.4, color=NOTE_COLORS_CV[hid], bg=False)

def process_ocarina(frame, app):
    global _hole_absent, _note_playing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    out = frame.copy()
    now = time.time()
    detected_ids = set(ids.flatten().tolist()) if ids is not None else set()

    hole_states = {}
    active_holes = []
    for hid in OCARINA_HOLE_IDS:
        if hid in detected_ids:
            _hole_absent[hid] = 0
            hole_states[hid] = False
        else:
            _hole_absent[hid] += 1
            covered = _hole_absent[hid] >= _ABSENT_THRESHOLD
            hole_states[hid] = covered
            if covered:
                last = _note_playing.get(hid, 0)
                if now - last > _NOTE_COOLDOWN:
                    _note_playing[hid] = now
                    _sound_cache[hid].stop()
                    _sound_cache[hid].play()
            if covered:
                active_holes.append(hid)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)

    body_idx = None
    if ids is not None:
        idlist = ids.flatten().tolist()
        if OCARINA_BODY_ID in idlist:
            body_idx = idlist.index(OCARINA_BODY_ID)
    if body_idx is not None:
        rvec,tvec = _pose_from_corners([corners[body_idx]])
        if tvec is not None:
            _draw_ocarina_body(out, corners[body_idx], rvec, tvec, hole_states)

    # Legenda buracos na imagem
    legend_y = frame.shape[0]-30
    for i, hid in enumerate(OCARINA_HOLE_IDS):
        col = NOTE_COLORS_CV[hid]
        x0 = 10+i*80
        cv2.circle(out, (x0, legend_y), 7, col, -1)
        put_text(out, f"{hid}={NOTE_NAMES[hid]}", (x0-4, legend_y+20),
                 scale=0.35, color=col, bg=False)

    if active_holes:
        names = "  ".join(NOTE_NAMES[h] for h in active_holes)
        put_text(out, f"Tocando: {names}", (10,30), scale=0.65, color=C_YEL)
        app.set_status(f"🎵 {names}")
        app.update_holes(active_holes)
    else:
        put_text(out, "Cubra os buracos (IDs 1–7) para tocar", (10,30),
                 scale=0.55, color=C_YEL)
        app.set_status("Aguardando cobertura de buraco…")
        app.update_holes([])

    return out

# ══════════════════════════════════════════════
# MODO 3 — AR SEM MARCADORES
# ══════════════════════════════════════════════
CUBE_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
              (0,4),(1,5),(2,6),(3,7)]
CUBE_FACES = [
    ([0,1,2,3],(80,120,220)),([4,5,6,7],(60,200,120)),
    ([0,1,5,4],(220,80,80)), ([3,2,6,7],(200,180,60)),
    ([0,3,7,4],(180,60,200)),([1,2,6,5],(60,200,220)),
]

def _build_cube(center, size):
    s = size/2
    verts = np.array([
        [-s,-s,-s],[s,-s,-s],[s,s,-s],[-s,s,-s],
        [-s,-s,s],[s,-s,s],[s,s,s],[-s,s,s],
    ], dtype=np.float32)
    return verts + center

def _project_point(pt3):
    fx=CAM_MTX[0,0]; fy=CAM_MTX[1,1]; cx_=CAM_MTX[0,2]; cy_=CAM_MTX[1,2]
    z=max(pt3[2],0.001)
    return (int(fx*pt3[0]/z+cx_), int(fy*pt3[1]/z+cy_))

def _draw_cube_on_frame(out, verts3d, alpha=0.6):
    pts2d = np.array([_project_point(v) for v in verts3d])
    h,w = out.shape[:2]
    mask = (pts2d[:,0]>=0)&(pts2d[:,0]<w)&(pts2d[:,1]>=0)&(pts2d[:,1]<h)
    if mask.sum() < 4:
        return
    overlay = out.copy()
    face_depths = sorted(
        [(np.mean([verts3d[i][2] for i in idxs]), fi)
         for fi,(idxs,col) in enumerate(CUBE_FACES)], reverse=True)
    for _, fi in face_depths:
        idxs,col = CUBE_FACES[fi]
        poly = pts2d[idxs]
        if poly.shape[0] >= 3:
            cv2.fillConvexPoly(overlay, poly, col)
    cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)
    for i,j in CUBE_EDGES:
        if mask[i] and mask[j]:
            cv2.line(out, tuple(pts2d[i]), tuple(pts2d[j]), (255,255,255), 1, cv2.LINE_AA)

def _estimate_palm_depth(lms, w, h):
    p0 = np.array([lms[0].x*w, lms[0].y*h], dtype=np.float32)
    p9 = np.array([lms[9].x*w, lms[9].y*h], dtype=np.float32)
    px_dist = float(np.linalg.norm(p9-p0))
    if px_dist < 1: return 0.5
    f = (CAM_MTX[0,0]+CAM_MTX[1,1])/2
    return 0.10 * f / px_dist

def process_markerless(frame, app):
    global _mp_frame_ts
    h,w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    _mp_frame_ts += 33
    result = _hands_model.detect_for_video(mp_image, _mp_frame_ts)
    out = frame.copy()

    if result.hand_landmarks:
        for hand_lm_list in result.hand_landmarks:
            lms = hand_lm_list
            for a,b in _HAND_CONNECTIONS:
                cv2.line(out,
                         (int(lms[a].x*w),int(lms[a].y*h)),
                         (int(lms[b].x*w),int(lms[b].y*h)),
                         (200,200,200),1,cv2.LINE_AA)
            for lm in lms:
                cv2.circle(out,(int(lm.x*w),int(lm.y*h)),3,(80,200,255),-1)

            palm_ids=[0,5,9,13,17]
            px = np.mean([lms[i].x for i in palm_ids])*w
            py = np.mean([lms[i].y for i in palm_ids])*h
            depth = _estimate_palm_depth(lms,w,h)
            fx=CAM_MTX[0,0]; fy=CAM_MTX[1,1]
            cx_c=CAM_MTX[0,2]; cy_c=CAM_MTX[1,2]
            X=(px-cx_c)*depth/fx; Y=(py-cy_c)*depth/fy
            palm_3d = np.array([X,Y,depth], dtype=np.float32)
            _draw_cube_on_frame(out, _build_cube(palm_3d, 0.06))
            cv2.circle(out,(int(px),int(py)),8,C_YEL,-1)
            put_text(out,f"Z~{depth*100:.0f}cm",(int(px)+12,int(py)),
                     scale=0.5,color=C_YEL)
            app.set_status(f"✋ Mão detectada | dist ~{depth*100:.0f} cm")
    else:
        put_text(out,"Mostre a palma da mão à câmera",(10,30),
                 scale=0.6,color=C_YEL)
        app.set_status("Aguardando mão…")

    return out

# ══════════════════════════════════════════════
# INTERFACE TKINTER
# ══════════════════════════════════════════════

DARK    = "#0e0e12"
PANEL   = "#1a1a24"
ACCENT  = "#4f8ef7"
ACCENT2 = "#a259ff"
TEXT_W  = "#e8e8f0"
TEXT_DIM= "#6b6b80"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visão Computacional — TP2")
        self.configure(bg=DARK)
        self.resizable(False, False)

        self.mode = tk.IntVar(value=1)
        self._running = True
        self._active_holes: list = []

        self._build_ui()

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.bind("<KeyPress>", self._on_key)
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self._update()

    # ── Layout ────────────────────────────────
    def _build_ui(self):
        # ---- Header ----
        header = tk.Frame(self, bg=PANEL, pady=8)
        header.pack(fill="x")
        tk.Label(header, text="👁  VISÃO COMPUTACIONAL",
                 font=("Courier New", 14, "bold"),
                 fg=ACCENT, bg=PANEL).pack(side="left", padx=14)
        tk.Label(header, text="TP2",
                 font=("Courier New", 10),
                 fg=ACCENT2, bg=PANEL).pack(side="left")
        tk.Label(header, text=CALIB_STATUS,
                 font=("Courier New", 9),
                 fg="#55cc88" if "carregada" in CALIB_STATUS else "#ffaa33",
                 bg=PANEL).pack(side="right", padx=14)

        # ---- Main row ----
        main = tk.Frame(self, bg=DARK)
        main.pack(fill="both", expand=True, padx=0, pady=0)

        # Camera canvas
        self.canvas = tk.Canvas(main, width=640, height=480,
                                bg="#000", highlightthickness=0)
        self.canvas.pack(side="left")

        # Side panel
        side = tk.Frame(main, bg=PANEL, width=220)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text="MODO", font=("Courier New", 9, "bold"),
                 fg=TEXT_DIM, bg=PANEL).pack(pady=(18,4))

        modes = [
            (1, "📏 Metrologia",  "Mede distância\nentre marcadores"),
            (2, "🎵 Ocarina",     "Toca notas ao\ncolocar dedos"),
            (3, "✋ AR Palma",    "Objeto 3D sobre\na palma da mão"),
        ]
        self._mode_btns = {}
        for val, label, tip in modes:
            f = tk.Frame(side, bg=PANEL)
            f.pack(fill="x", padx=10, pady=4)
            btn = tk.Button(
                f, text=label,
                font=("Courier New", 10, "bold"),
                fg=TEXT_W, bg="#252535",
                activebackground=ACCENT, activeforeground="#fff",
                relief="flat", cursor="hand2", pady=8,
                command=lambda v=val: self._set_mode(v)
            )
            btn.pack(fill="x")
            tk.Label(f, text=tip, font=("Courier New", 7),
                     fg=TEXT_DIM, bg=PANEL, justify="left").pack(anchor="w", padx=4)
            self._mode_btns[val] = btn

        # Separator
        tk.Frame(side, bg="#2a2a3a", height=1).pack(fill="x", padx=10, pady=10)

        # Status
        tk.Label(side, text="STATUS", font=("Courier New", 9, "bold"),
                 fg=TEXT_DIM, bg=PANEL).pack()
        self._status_var = tk.StringVar(value="Iniciando…")
        tk.Label(side, textvariable=self._status_var,
                 font=("Courier New", 9), fg=TEXT_W, bg=PANEL,
                 wraplength=190, justify="left").pack(padx=10, pady=6)

        # Separator
        tk.Frame(side, bg="#2a2a3a", height=1).pack(fill="x", padx=10, pady=10)

        # Ocarina note display
        tk.Label(side, text="NOTAS", font=("Courier New", 9, "bold"),
                 fg=TEXT_DIM, bg=PANEL).pack()
        self._note_frame = tk.Frame(side, bg=PANEL)
        self._note_frame.pack(pady=6, padx=8, fill="x")
        self._note_labels: dict[int, tk.Label] = {}
        for i, (hid, name) in enumerate(NOTE_NAMES.items()):
            col = NOTE_COLORS_HEX[hid]
            lbl = tk.Label(self._note_frame, text=f" {name} ",
                           font=("Courier New", 9, "bold"),
                           fg="#222", bg="#333",
                           relief="flat", padx=4, pady=2)
            lbl.grid(row=i//4, column=i%4, padx=2, pady=2, sticky="ew")
            self._note_labels[hid] = lbl

        # Spacer + keys legend
        tk.Frame(side, bg=PANEL).pack(expand=True, fill="y")
        tk.Frame(side, bg="#2a2a3a", height=1).pack(fill="x", padx=10, pady=4)
        keys_info = "1 / 2 / 3 — trocar modo\nQ  /  Esc — sair"
        tk.Label(side, text=keys_info, font=("Courier New", 8),
                 fg=TEXT_DIM, bg=PANEL, justify="left").pack(padx=10, pady=(0,14))

        # Quit button
        tk.Button(side, text="⏹  SAIR",
                  font=("Courier New", 9, "bold"),
                  fg="#ff6060", bg="#1a1a24",
                  activebackground="#ff6060", activeforeground="#fff",
                  relief="flat", cursor="hand2", pady=6,
                  command=self.quit_app).pack(fill="x", padx=10, pady=(0,12))

        self._refresh_mode_buttons()

    # ── Helpers ───────────────────────────────
    def _set_mode(self, val):
        self.mode.set(val)
        if val == 1:
            _metro_history.clear()
        self.update_holes([])
        self._refresh_mode_buttons()

    def _refresh_mode_buttons(self):
        cur = self.mode.get()
        for val, btn in self._mode_btns.items():
            if val == cur:
                btn.config(bg=ACCENT, fg="#fff")
            else:
                btn.config(bg="#252535", fg=TEXT_W)

    def set_status(self, text: str):
        self._status_var.set(text)

    def update_holes(self, active: list):
        self._active_holes = active
        for hid, lbl in self._note_labels.items():
            if hid in active:
                lbl.config(bg=NOTE_COLORS_HEX[hid], fg="#111")
            else:
                lbl.config(bg="#333", fg="#555")

    def _on_key(self, event):
        k = event.keysym.lower()
        if k in ("q", "escape"):
            self.quit_app()
        elif k == "1":
            self._set_mode(1)
        elif k == "2":
            self._set_mode(2)
        elif k == "3":
            self._set_mode(3)

    # ── Camera loop ───────────────────────────
    def _update(self):
        if not self._running:
            return
        ret, frame = self.cap.read()
        if ret:
            mode = self.mode.get()
            processors = {1: process_metrology, 2: process_ocarina, 3: process_markerless}
            out = processors[mode](frame, self)

            # Convert BGR → RGB → PIL → ImageTk
            img_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            self._imgtk = ImageTk.PhotoImage(image=img_pil)
            self.canvas.create_image(0, 0, anchor="nw", image=self._imgtk)

        self.after(16, self._update)   # ~60 fps target

    def quit_app(self):
        self._running = False
        self.cap.release()
        pygame.mixer.quit()
        self.destroy()


# ══════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()