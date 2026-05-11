import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
import threading
import sys
import os
import math
import time
import struct
import wave
import tempfile

# ─────────────────────────────────────────────────────────────────────────────
# ÁUDIO – geração de tons sem dependências externas (wave + winsound / pyaudio)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_wav_bytes(frequency: float, duration: float = 0.35,
                         sample_rate: int = 44100, volume: float = 0.5) -> bytes:
    """Gera um arquivo WAV em memória para a frequência dada."""
    n_samples = int(sample_rate * duration)
    buf = bytearray()
    for i in range(n_samples):
        t = i / sample_rate
        # tom com envelope ADSR simples
        envelope = min(1.0, i / (sample_rate * 0.01))          # attack
        envelope *= min(1.0, (n_samples - i) / (sample_rate * 0.05))  # release
        sample = int(32767 * volume * envelope * math.sin(2 * math.pi * frequency * t))
        buf += struct.pack('<h', sample)
    # monta WAV
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        fname = f.name
    with wave.open(fname, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(buf))
    return fname


def play_tone(frequency: float):
    """Toca um tom em thread separada (não bloqueia o loop de vídeo)."""
    def _play():
        fname = _generate_wav_bytes(frequency)
        try:
            if sys.platform == 'win32':
                import winsound
                winsound.PlaySound(fname, winsound.SND_FILENAME)
            else:
                os.system(f'aplay -q "{fname}" 2>/dev/null || '
                          f'afplay "{fname}" 2>/dev/null || '
                          f'ffplay -nodisp -autoexit -loglevel quiet "{fname}" 2>/dev/null')
        finally:
            try:
                os.unlink(fname)
            except Exception:
                pass
    threading.Thread(target=_play, daemon=True).start()


# Notas da escala pentatônica de Dó (oito orifícios da ocarina clássica)
OCARINA_NOTES = {
    0: ('C4',  261.63),
    1: ('D4',  293.66),
    2: ('E4',  329.63),
    3: ('G4',  392.00),
    4: ('A4',  440.00),
    5: ('C5',  523.25),
    6: ('D5',  587.33),
    7: ('E5',  659.25),
}

# ─────────────────────────────────────────────────────────────────────────────
# CÂMERA
# ─────────────────────────────────────────────────────────────────────────────
def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera {index}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    return cap


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR ArUco
# ─────────────────────────────────────────────────────────────────────────────
ARUCO_DICT  = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters()
ARUCO_DETECTOR = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)


def detect_aruco(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    return corners, ids


def marker_center(corners_single):
    """Retorna o centro (x, y) de um marcador."""
    c = corners_single[0]
    return (int(c[:, 0].mean()), int(c[:, 1].mean()))


def marker_size_px(corners_single):
    """Retorna o tamanho médio do marcador em pixels."""
    c = corners_single[0]
    w = np.linalg.norm(c[0] - c[1])
    h = np.linalg.norm(c[1] - c[2])
    return (w + h) / 2


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO (3a) – METROLOGIA
# ─────────────────────────────────────────────────────────────────────────────
MARKER_REAL_CM = 4.0          # tamanho REAL do marcador em cm (ajuste conforme necessário)
REF_MARKER_IDS = [0, 1]       # IDs dos dois marcadores de referência


def run_metrologia(cap: cv2.VideoCapture):
    print("\n[Metrologia] Posicione os marcadores ArUco ID=0 e ID=1 em frente à câmera.")
    print("  Defina o tamanho real do marcador: ", end='')
    try:
        size_input = float(input(f"({MARKER_REAL_CM} cm) > ").strip() or MARKER_REAL_CM)
    except ValueError:
        size_input = MARKER_REAL_CM

    win = "Metrologia ArUco | Q = sair"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        corners, ids = detect_aruco(frame)
        aruco.drawDetectedMarkers(frame, corners, ids)

        info_lines = ["IDs detectados: " + (str(ids.flatten().tolist()) if ids is not None else "nenhum")]

        if ids is not None and len(ids) >= 2:
            id_map = {int(ids[i]): corners[i] for i in range(len(ids))}
            if REF_MARKER_IDS[0] in id_map and REF_MARKER_IDS[1] in id_map:
                c0 = id_map[REF_MARKER_IDS[0]]
                c1 = id_map[REF_MARKER_IDS[1]]

                # calibração de escala px/cm via tamanho do marcador
                px_per_cm = marker_size_px(c0) / size_input

                center0 = marker_center(c0)
                center1 = marker_center(c1)
                dist_px = math.dist(center0, center1)
                dist_cm = dist_px / px_per_cm

                cv2.line(frame, center0, center1, (0, 255, 0), 2)
                mid = ((center0[0]+center1[0])//2, (center0[1]+center1[1])//2)
                cv2.putText(frame, f"{dist_cm:.1f} cm", (mid[0]+10, mid[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                info_lines.append(f"Distancia ID0-ID1: {dist_cm:.1f} cm  ({dist_px:.0f} px)")

        # HUD
        for i, txt in enumerate(info_lines):
            cv2.putText(frame, txt, (10, 30 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyWindow(win)


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO (3b) – OCARINA
# ─────────────────────────────────────────────────────────────────────────────
# Mapeamento: ID 10 = corpo/referência da ocarina; IDs 11-18 = orifícios
OCARINA_REF_ID   = 10
OCARINA_HOLE_IDS = [11, 12, 13, 14, 15, 16, 17, 18]   # até 8 orifícios

COOLDOWN_S = 0.4   # intervalo mínimo entre sons do mesmo orifício


def _draw_ocarina_3d(frame, ref_corners, holes_covered: dict):
    """
    Desenha uma ocarina 3D simplificada sobre o marcador de referência.
    A posição/escala segue o marcador ArUco de referência.
    """
    c = ref_corners[0]   # 4 pontos do marcador

    # Vetores de perspectiva a partir dos cantos do marcador
    tl = c[0].astype(float)
    tr = c[1].astype(float)
    br = c[2].astype(float)
    bl = c[3].astype(float)

    def lerp(a, b, t):
        return a + t * (b - a)

    # ----- corpo da ocarina (elipse perspectivada via polígono) -----
    body_pts = []
    for t in np.linspace(0, 1, 40):
        angle = 2 * math.pi * t
        u = 0.5 + 1.4 * math.cos(angle)   # exagerado para parecer ovalado
        v = 0.5 + 0.55 * math.sin(angle)
        u = max(0, min(1, u)) if False else u   # sem clip – perspectiva livre
        pt = lerp(lerp(tl, tr, u), lerp(bl, br, u), v)
        body_pts.append(pt)
    body_pts = np.array(body_pts, dtype=np.int32)

    # sombra / profundidade
    shadow = body_pts + np.array([6, 6])
    cv2.fillPoly(frame, [shadow], (30, 20, 10))
    # corpo principal – gradiente simulado com dois polígonos
    cv2.fillPoly(frame, [body_pts], (30, 80, 160))
    # brilho superior
    highlight_pts = []
    for t in np.linspace(0, 1, 30):
        angle = math.pi + math.pi * t  # metade superior
        u = 0.5 + 1.3 * math.cos(angle)
        v = 0.5 + 0.45 * math.sin(angle)
        pt = lerp(lerp(tl, tr, u), lerp(bl, br, u), v)
        highlight_pts.append(pt)
    cv2.polylines(frame, [np.array(highlight_pts, dtype=np.int32)], False, (120, 180, 255), 3)

    # bocal (tubo pequeno à direita)
    mouthpiece_start = lerp(lerp(tl, tr, 1.5), lerp(bl, br, 1.5), 0.5)
    mouthpiece_end   = lerp(lerp(tl, tr, 1.9), lerp(bl, br, 1.9), 0.5)
    cv2.line(frame,
             tuple(mouthpiece_start.astype(int)),
             tuple(mouthpiece_end.astype(int)),
             (20, 60, 130), 12)
    cv2.circle(frame, tuple(mouthpiece_end.astype(int)), 6, (100, 160, 220), -1)

    # ----- orifícios -----
    n_holes = len(OCARINA_HOLE_IDS)
    for i in range(n_holes):
        t_u = 0.2 + 0.6 * (i / max(n_holes - 1, 1))
        t_v = 0.5
        hole_center = lerp(lerp(tl, tr, t_u), lerp(bl, br, t_u), t_v)
        hid = OCARINA_HOLE_IDS[i]
        covered = holes_covered.get(hid, False)
        color   = (0, 20, 60) if covered else (180, 220, 255)
        cv2.circle(frame, tuple(hole_center.astype(int)), 10, color, -1)
        cv2.circle(frame, tuple(hole_center.astype(int)), 10, (0, 40, 100), 2)
        # nome da nota
        note_name = OCARINA_NOTES.get(i, ('?', 0))[0]
        cv2.putText(frame, note_name,
                    (int(hole_center[0]) - 12, int(hole_center[1]) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 255), 1)


def run_ocarina(cap: cv2.VideoCapture):
    print("\n[Ocarina] Marcadores necessários:")
    print(f"  ID {OCARINA_REF_ID} = corpo da ocarina (referência)")
    for i, hid in enumerate(OCARINA_HOLE_IDS):
        note = OCARINA_NOTES[i][0]
        print(f"  ID {hid} = orifício {i+1} → nota {note}")
    print("Cubra cada marcador de orifício com o dedo para tocar a nota.")
    print("Q = sair\n")

    win = "Ocarina Virtual | Q = sair"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_played  = {}   # hid → timestamp
    was_covered  = {}   # hid → bool (estado anterior)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        corners, ids = detect_aruco(frame)

        id_map = {}
        if ids is not None:
            for i in range(len(ids)):
                id_map[int(ids[i])] = corners[i]

        holes_covered = {}
        for hid in OCARINA_HOLE_IDS:
            holes_covered[hid] = (hid not in id_map)   # coberto = não detectado

        # Desenha ocarina se marcador de referência visível
        if OCARINA_REF_ID in id_map:
            _draw_ocarina_3d(frame, id_map[OCARINA_REF_ID], holes_covered)
        else:
            cv2.putText(frame, f"Mostre marcador ID={OCARINA_REF_ID} (corpo)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # Toca sons ao cobrir orifícios
        now = time.time()
        for i, hid in enumerate(OCARINA_HOLE_IDS):
            covered = holes_covered[hid]
            prev    = was_covered.get(hid, False)
            if covered and not prev:   # borda de descida (acabou de cobrir)
                last_t = last_played.get(hid, 0)
                if now - last_t > COOLDOWN_S:
                    freq = OCARINA_NOTES[i][1]
                    play_tone(freq)
                    last_played[hid] = now
            was_covered[hid] = covered

        aruco.drawDetectedMarkers(frame, corners, ids)

        # HUD de status dos orifícios
        for i, hid in enumerate(OCARINA_HOLE_IDS):
            note  = OCARINA_NOTES[i][0]
            state = "●" if holes_covered[hid] else "○"
            color = (0, 80, 255) if holes_covered[hid] else (180, 180, 180)
            cv2.putText(frame, f"{state} {note}", (10, 30 + 28*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyWindow(win)


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO (3c) – REALIDADE VIRTUAL SEM MARCADORES (MediaPipe + objeto 3D)
# ─────────────────────────────────────────────────────────────────────────────

# Compatibilidade: tenta API nova (0.10+), cai para API legada se necessário
try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import HandLandmarkerOptions
    _MP_NEW_API = True
except Exception:
    _MP_NEW_API = False

# Conexões dos dedos para desenhar o esqueleto manualmente
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # polegar
    (0,5),(5,6),(6,7),(7,8),       # indicador
    (0,9),(9,10),(10,11),(11,12),  # médio
    (0,13),(13,14),(14,15),(15,16),# anelar
    (0,17),(17,18),(18,19),(19,20),# mínimo
    (5,9),(9,13),(13,17),          # metacarpo
]


def _draw_hand_skeleton(frame, landmarks, w, h):
    """Desenha o esqueleto da mão a partir de uma lista de NormalizedLandmark."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 120, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 200, 100), -1)


def _palm_center_from_list(landmarks, w, h):
    palm_ids = [0, 1, 5, 9, 13, 17]
    xs = [landmarks[i].x for i in palm_ids]
    ys = [landmarks[i].y for i in palm_ids]
    return int(np.mean(xs) * w), int(np.mean(ys) * h)


def _palm_radius_from_list(landmarks, w, h):
    wrist = landmarks[0]
    mid   = landmarks[9]
    dx = (mid.x - wrist.x) * w
    dy = (mid.y - wrist.y) * h
    return max(20, int(math.hypot(dx, dy) * 0.6))


def _draw_3d_sphere(frame, cx, cy, radius, t):
    """
    Desenha uma esfera 3D animada com sombreamento Phong simplificado.
    t = tempo para animação de rotação.
    """
    if radius < 5:
        return

    # Sombra no chão
    shadow_a = int(radius * 1.3)
    shadow_b = int(radius * 0.4)
    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, cy + radius + shadow_b),
                (shadow_a, shadow_b), 0, 0, 360, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Corpo da esfera (gradiente radial simulado com círculos concêntricos)
    steps = 20
    for s in range(steps, 0, -1):
        ratio = s / steps
        r_step = int(radius * ratio)
        # interpolação de cor: centro brilhante → borda escura
        # cor base: azul-ciano animado
        hue = (int(t * 30) + int(ratio * 60)) % 180
        hsv_color = np.uint8([[[hue, 220, int(255 * ratio)]]])
        bgr = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        # offset de luz (simula fonte de luz no canto superior esquerdo)
        lx = cx - int(radius * 0.3)
        ly = cy - int(radius * 0.3)
        cv2.circle(frame, (lx + int((cx-lx)*(1-ratio)),
                           ly + int((cy-ly)*(1-ratio))),
                   r_step, color, -1)

    # Especular
    spec_x = cx - int(radius * 0.35)
    spec_y = cy - int(radius * 0.35)
    cv2.circle(frame, (spec_x, spec_y), max(2, radius // 6), (240, 240, 255), -1)
    cv2.circle(frame, (spec_x - 4, spec_y - 4), max(1, radius // 12), (255, 255, 255), -1)

    # Anel orbital animado (rotação)
    angle_deg = (t * 90) % 360
    ring_pts = []
    for deg in range(0, 360, 10):
        rad = math.radians(deg)
        rx  = math.cos(rad) * radius
        ry  = math.sin(rad) * radius * 0.3
        # rotação
        a2  = math.radians(angle_deg)
        rx2 = rx * math.cos(a2) - ry * math.sin(a2)
        ry2 = rx * math.sin(a2) + ry * math.cos(a2)
        ring_pts.append((cx + int(rx2), cy + int(ry2 * 0.5)))
    ring_pts = np.array(ring_pts, dtype=np.int32)
    cv2.polylines(frame, [ring_pts], True, (0, 220, 255), 2)

    # Segundo anel (perpendicular)
    ring2 = []
    for deg in range(0, 360, 10):
        rad = math.radians(deg)
        rx  = math.cos(rad) * radius * 0.3
        ry  = math.sin(rad) * radius
        a2  = math.radians(angle_deg + 90)
        rx2 = rx * math.cos(a2) - ry * math.sin(a2)
        ry2 = rx * math.sin(a2) + ry * math.cos(a2)
        ring2.append((cx + int(rx2 * 0.5), cy + int(ry2)))
    ring2 = np.array(ring2, dtype=np.int32)
    cv2.polylines(frame, [ring2], True, (255, 180, 0), 2)


def _run_ar_hand_legacy(cap, win):
    """Usa mediapipe.solutions.hands (API legada, versões < 0.10)."""
    mp_hands_mod = mp.solutions.hands
    mp_drawing   = mp.solutions.drawing_utils

    with mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            t     = time.time() - t0

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hl, mp_hands_mod.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,200,100), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0,120,255), thickness=2),
                    )
                    lms = hl.landmark
                    cx, cy = _palm_center_from_list(lms, w, h)
                    radius = _palm_radius_from_list(lms, w, h)
                    _draw_3d_sphere(frame, cx, cy, radius, t)
            else:
                cv2.putText(frame, "Mostre sua mao para a camera",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)

            cv2.putText(frame, "AR Sem Marcadores – Esfera 3D na Palma",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,220,0), 2)
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break


def _run_ar_hand_new(cap, win):
    """Usa mediapipe.tasks (API nova, versões >= 0.10)."""
    import mediapipe as mp2
    BaseOptions   = mp2.tasks.BaseOptions
    HandLandmarker        = mp2.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp2.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp2.tasks.vision.RunningMode

    # Baixa o modelo se necessário
    model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("  Baixando modelo HandLandmarker (~8 MB)...", end=' ', flush=True)
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        try:
            urllib.request.urlretrieve(url, model_path)
            print("OK")
        except Exception as e:
            print(f"\n  [ERRO] Falha ao baixar modelo: {e}")
            print("  Certifique-se de ter conexão com a internet e tente novamente.")
            return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    t0 = time.time()
    with HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            t     = time.time() - t0

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp2.Image(image_format=mp2.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            if result.hand_landmarks:
                for lms in result.hand_landmarks:
                    _draw_hand_skeleton(frame, lms, w, h)
                    cx, cy = _palm_center_from_list(lms, w, h)
                    radius = _palm_radius_from_list(lms, w, h)
                    _draw_3d_sphere(frame, cx, cy, radius, t)
            else:
                cv2.putText(frame, "Mostre sua mao para a camera",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)

            cv2.putText(frame, "AR Sem Marcadores – Esfera 3D na Palma",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,220,0), 2)
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break


def run_ar_hand(cap: cv2.VideoCapture):
    print("\n[AR sem marcadores] Mostre sua mão para a câmera.")
    print("O objeto 3D aparecerá no centro da palma. Q = sair\n")

    win = "AR Mão | Q = sair"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Detecta qual API do MediaPipe está disponível
    has_solutions = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')
    if has_solutions:
        print("  [MediaPipe] Usando API legada (mp.solutions.hands)")
        _run_ar_hand_legacy(cap, win)
    else:
        print("  [MediaPipe] Usando API nova (mediapipe.tasks)")
        _run_ar_hand_new(cap, win)

    cv2.destroyWindow(win)


# ─────────────────────────────────────────────────────────────────────────────
# MENU PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║     TRABALHO PRÁTICO 2 – VISÃO COMPUTACIONAL             ║
║     UNICENTRO  •  Prof. Dr. Mauro Miazaki                ║
╠══════════════════════════════════════════════════════════╣
║  [1]  Metrologia ArUco  (distância entre marcadores)     ║
║  [2]  Ocarina Virtual   (tocar notas cobrindo ArUco)     ║
║  [3]  AR sem marcadores (esfera 3D na palma – MediaPipe) ║
║  [0]  Sair                                               ║
╚══════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)
    print("Abrindo câmera...", end=' ', flush=True)
    cap = open_camera(0)
    print("OK")

    while True:
        print(BANNER)
        choice = input("Escolha uma opção: ").strip()

        if choice == '1':
            run_metrologia(cap)
        elif choice == '2':
            run_ocarina(cap)
        elif choice == '3':
            run_ar_hand(cap)
        elif choice == '0':
            break
        else:
            print("Opção inválida. Tente novamente.")

    cap.release()
    cv2.destroyAllWindows()
    print("Até logo!")


if __name__ == '__main__':
    main()