"""
=============================================================================
3º Trabalho Prático de Visão Computacional – UNICENTRO
Análise 3D de Raízes com PyVista
Prof. Dr. Mauro Miazaki
=============================================================================
Requisitos implementados:
  (1) Interface interativa (tkinter)
  (2) Carregamento e pré-processamento (limpeza de bordas)
  (3a) DVR – Direct Volume Rendering
  (3b) Isosuperfície com suavização e decimação
  (3c) Esqueleto + visualização combinada isosuperfície + esqueleto
  (3d) Métricas: volume, área, compacidade, excentricidade, 5+ métricas de esqueleto
  (3e) Visualização dividida e sincronizada (DVR | Isosuperfície)
  (3f) Relatório em texto

Uso:
  python trabalho3_visao_computacional.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import io
import zipfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from PIL import Image
import pyvista as pv
from skimage import filters, morphology
from skimage.morphology import skeletonize
from scipy import ndimage

# ---------------------------------------------------------------------------
# Constantes / configurações globais
# ---------------------------------------------------------------------------
BORDER_CLEAN = 5    # pixels a zerar nas bordas externas de cada fatia
ISO_VALUE    = 0.5  # valor padrão da isosuperfície (0..1 após normalização)
SMOOTH_ITER  = 50   # iterações de suavização Laplaciana
DECIMATE_T   = 0.70 # fração de decimação (0..1)

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def load_images_from_zip(zip_path: str) -> np.ndarray:
    """Extrai e empilha todas as imagens PNG de um ZIP em um volume 3-D uint8."""
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".png"))
        if not names:
            raise ValueError(f"Nenhuma imagem PNG encontrada em {zip_path}")
        slices = []
        for name in names:
            data = z.read(name)
            img  = Image.open(io.BytesIO(data)).convert("L")
            slices.append(np.array(img, dtype=np.uint8))
    return np.stack(slices, axis=0)   # shape: (Z, Y, X)


def clean_borders(volume: np.ndarray, border: int = BORDER_CLEAN) -> np.ndarray:
    """Zera os pixels das bordas externas de cada fatia (XY) e dos extremos Z."""
    vol = volume.copy()
    b = border
    # bordas laterais (XY) – todas as fatias Z
    vol[:, :b,  :]  = 0
    vol[:, -b:, :]  = 0
    vol[:, :,   :b] = 0
    vol[:, :,  -b:] = 0
    # bordas Z (primeiras e últimas fatias)
    vol[:b,  :, :] = 0
    vol[-b:, :, :] = 0
    return vol


def volume_to_pyvista(volume: np.ndarray) -> pv.ImageData:
    """Converte ndarray (Z,Y,X) para pv.ImageData com escalares normalizados [0,1]."""
    grid = pv.ImageData()
    grid.dimensions = (volume.shape[2], volume.shape[1], volume.shape[0])
    flat = volume.transpose(2, 1, 0).ravel(order="F").astype(np.float32)
    mx = flat.max()
    flat = flat / mx if mx > 0 else flat
    grid.point_data["scalars"] = flat
    return grid


def skeleton_to_lines(skel_binary: np.ndarray) -> pv.PolyData:
    """
    Converte um esqueleto binário 3D em pv.PolyData com células de linha,
    conectando voxels vizinhos no esqueleto (26-conectividade).
    """
    coords = np.argwhere(skel_binary).astype(np.int32)   # (N,3) em Z,Y,X
    if len(coords) < 2:
        return pv.PolyData()

    # Índice reverso: posição → índice no array coords
    coord_set = {tuple(c): i for i, c in enumerate(coords)}

    # Para cada voxel, conectar com vizinhos em Z+, Y+, X+ (evita duplicatas)
    segments = []
    offsets = [
        (0, 0, 1), (0, 1, 0), (1, 0, 0),
        (0, 1, 1), (1, 0, 1), (1, 1, 0),
        (1, 1, 1), (0, 1, -1), (1, 0, -1),
        (1, -1, 0), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    ]
    for c in coords:
        for dz, dy, dx in offsets:
            nb = (c[0]+dz, c[1]+dy, c[2]+dx)
            if nb in coord_set:
                segments.append((coord_set[tuple(c)], coord_set[nb]))

    # Pontos em ordem X,Y,Z para PyVista
    points_xyz = coords[:, [2, 1, 0]].astype(float)
    cloud = pv.PolyData()
    cloud.points = points_xyz

    if segments:
        # Células de linha: [2, i, j]
        cells = np.array([[2, a, b] for a, b in segments], dtype=np.int32).ravel()
        cloud.lines = cells

    return cloud


# ---------------------------------------------------------------------------
# Pipeline de processamento
# ---------------------------------------------------------------------------

class RootData:
    """Armazena dados processados de uma raiz."""

    def __init__(self, name: str, zip_path: str, log_fn=None):
        self.name     = name
        self.zip_path = zip_path
        self._log     = log_fn or print

        self.volume_raw   = None   # uint8
        self.volume_clean = None   # uint8, bordas limpas
        self.grid         = None   # pv.ImageData
        self.iso_mesh     = None   # isosuperfície bruta
        self.iso_smooth   = None   # isosuperfície suavizada
        self.iso_decim    = None   # isosuperfície decimada
        self.skel_lines   = None   # pv.PolyData com linhas do esqueleto
        self.metrics      = {}
        self._binary      = None   # volume binário
        self._skel_binary = None   # esqueleto binário

    # -----------------------------------------------------------------------
    def load(self):
        self._log(f"[{self.name}] Carregando imagens de {os.path.basename(self.zip_path)} ...")
        self.volume_raw   = load_images_from_zip(self.zip_path)
        self.volume_clean = clean_borders(self.volume_raw, BORDER_CLEAN)
        z, y, x = self.volume_clean.shape
        self._log(f"[{self.name}] Volume: {x}×{y}×{z} voxels — bordas limpas ({BORDER_CLEAN}px)")
        self.grid = volume_to_pyvista(self.volume_clean)

    # -----------------------------------------------------------------------
    def compute_isosurface(self, iso_val: float = ISO_VALUE,
                           smooth_iter: int = SMOOTH_ITER,
                           decimate_target: float = DECIMATE_T):
        self._log(f"[{self.name}] Gerando isosuperfície (iso={iso_val:.2f}) ...")
        self.iso_mesh = self.grid.contour([iso_val], scalars="scalars")
        n_raw = self.iso_mesh.n_cells
        if n_raw == 0:
            raise RuntimeError(f"Isosuperfície vazia para iso={iso_val:.2f}. "
                               "Tente um valor menor.")
        self._log(f"[{self.name}] Isosuperfície bruta: {n_raw} faces")

        self._log(f"[{self.name}] Suavizando (iters={smooth_iter}) ...")
        self.iso_smooth = self.iso_mesh.smooth(
            n_iter=smooth_iter, relaxation_factor=0.1, boundary_smoothing=True
        )

        self._log(f"[{self.name}] Decimando ({decimate_target*100:.0f}% redução) ...")
        self.iso_decim = self.iso_smooth.decimate(decimate_target)
        n_dec = self.iso_decim.n_cells
        self._log(f"[{self.name}] Isosuperfície decimada: {n_dec} faces "
                  f"({100*n_dec/n_raw:.1f}% do original)")

    # -----------------------------------------------------------------------
    def compute_skeleton(self):
        self._log(f"[{self.name}] Computando esqueleto 3D (skimage) ...")
        vol_norm = self.volume_clean.astype(np.float32) / 255.0

        # Binarização com Otsu (apenas pixels não-nulos)
        nonzero = vol_norm[vol_norm > 0]
        if len(nonzero) == 0:
            self._log(f"[{self.name}] AVISO: volume zerado, esqueleto vazio.")
            self._binary = np.zeros_like(vol_norm, dtype=bool)
            self._skel_binary = np.zeros_like(vol_norm, dtype=bool)
            self.skel_lines = pv.PolyData()
            return

        thresh = filters.threshold_otsu(nonzero)
        binary = vol_norm > thresh

        # Limpeza morfológica
        binary = morphology.remove_small_objects(binary, min_size=64)
        binary = morphology.binary_closing(binary, morphology.ball(1))
        self._binary = binary

        # Skeletonize 3D
        skel = skeletonize(binary).astype(bool)
        self._skel_binary = skel
        n_skel = int(skel.sum())
        self._log(f"[{self.name}] Esqueleto: {n_skel} voxels")

        # Converter para linhas PyVista
        self.skel_lines = skeleton_to_lines(skel)
        self._log(f"[{self.name}] Esqueleto convertido para {self.skel_lines.n_cells} "
                  f"segmentos de linha.")

    # -----------------------------------------------------------------------
    def compute_metrics(self):
        self._log(f"[{self.name}] Calculando métricas ...")

        if self._binary is None or self._skel_binary is None:
            raise RuntimeError("Execute compute_skeleton() antes de compute_metrics().")

        mesh  = self.iso_decim
        skel  = self._skel_binary
        binary = self._binary

        # --- Volume (voxels binários)
        vol_voxels = float(binary.sum())
        self.metrics["volume_voxels"] = vol_voxels

        # --- Área de superfície (soma das áreas das faces da malha)
        surf_area = float(mesh.area)
        self.metrics["area_superficie"] = surf_area

        # --- Compacidade = (36π V²) / A³  →  1 para esfera perfeita
        if surf_area > 0:
            compacity = (36.0 * np.pi * vol_voxels**2) / (surf_area**3)
        else:
            compacity = 0.0
        self.metrics["compacidade"] = float(compacity)

        # --- Excentricidade via autovalores da matriz de covariância dos voxels
        coords = np.argwhere(binary).astype(float)
        if len(coords) > 3:
            centroid = coords.mean(axis=0)
            diff     = coords - centroid
            cov      = (diff.T @ diff) / len(coords)
            eigvals  = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
            eccentricity = float(np.sqrt(1.0 - eigvals[-1] / eigvals[0])) \
                           if eigvals[0] > 0 else 0.0
        else:
            eccentricity = 0.0
        self.metrics["excentricidade"] = eccentricity

        # ---- Métricas de esqueleto ----

        # Contagem de vizinhos 26-conectados no esqueleto
        struct = ndimage.generate_binary_structure(3, 3)
        nb_count = ndimage.convolve(
            skel.astype(np.int32), struct.astype(np.int32)
        )
        # nb_count inclui o próprio voxel, então vizinhos = nb_count - 1

        # 1) Comprimento total (nº voxels do esqueleto)
        self.metrics["skel_total_voxels"] = int(skel.sum())

        # 2) Pontos de ramificação: ≥ 3 vizinhos dentro do esqueleto
        branch_pts = (nb_count >= 4) & skel   # ≥3 vizinhos + ele mesmo = 4
        self.metrics["skel_branch_points"] = int(branch_pts.sum())

        # 3) Pontos terminais: exatamente 1 vizinho dentro do esqueleto
        tip_pts = (nb_count == 2) & skel      # 1 vizinho + ele mesmo = 2
        self.metrics["skel_tip_points"] = int(tip_pts.sum())

        # 4) Densidade do esqueleto = voxels_skel / voxels_binário
        self.metrics["skel_density"] = float(skel.sum()) / max(float(binary.sum()), 1)

        # 5) Alongamento (bounding-box ratio: dim_max / dim_min)
        z_idx, y_idx, x_idx = np.where(skel)
        if len(z_idx) > 0:
            dz = int(z_idx.max() - z_idx.min() + 1)
            dy = int(y_idx.max() - y_idx.min() + 1)
            dx = int(x_idx.max() - x_idx.min() + 1)
            elongation = float(max(dz, dy, dx)) / float(max(min(dz, dy, dx), 1))
            diag = float(np.sqrt(dz**2 + dy**2 + dx**2))
        else:
            elongation = 0.0
            diag = 1.0
        self.metrics["skel_elongation"] = elongation

        # 6) Tortuosidade = comprimento_skel / diagonal_bounding_box
        self.metrics["skel_tortuosity"] = float(skel.sum()) / max(diag, 1.0)

        # 7) Número de segmentos (ramos entre pontos de ramificação/terminais)
        skel_no_branch = skel.copy()
        skel_no_branch[branch_pts] = False
        _, n_segs = ndimage.label(skel_no_branch, structure=struct)
        self.metrics["skel_segments"] = int(n_segs)

        self._log(f"[{self.name}] Métricas calculadas com sucesso.")

    # -----------------------------------------------------------------------
    def run_full_pipeline(self, iso_val=ISO_VALUE,
                          smooth_iter=SMOOTH_ITER,
                          decimate_target=DECIMATE_T):
        self.load()
        self.compute_isosurface(iso_val, smooth_iter, decimate_target)
        self.compute_skeleton()
        self.compute_metrics()


# ---------------------------------------------------------------------------
# GUI principal
# ---------------------------------------------------------------------------

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Visão Computacional – Análise 3D de Raízes")
        self.geometry("1100x720")
        self.configure(bg="#1e1e2e")

        self.roots: dict = {}
        self.zip_b0207 = tk.StringVar()
        self.zip_b0309 = tk.StringVar()
        self.iso_val   = tk.DoubleVar(value=ISO_VALUE)
        self.smooth_it = tk.IntVar(value=SMOOTH_ITER)
        self.decim_t   = tk.DoubleVar(value=DECIMATE_T)

        self._build_ui()

    # -----------------------------------------------------------------------
    def _build_ui(self):
        # ---- estilos -------------------------------------------------------
        style = ttk.Style(self)
        style.theme_use("clam")
        BG   = "#1e1e2e"
        FG   = "#cdd6f4"
        ABTN = "#89b4fa"

        style.configure("TFrame",      background=BG)
        style.configure("TLabel",      background=BG, foreground=FG,
                                       font=("Segoe UI", 10))
        style.configure("TLabelframe", background=BG, foreground=ABTN,
                                       font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background=BG, foreground=ABTN)
        style.configure("TButton",     background="#313244", foreground=FG,
                                       font=("Segoe UI", 10), relief="flat",
                                       padding=(8, 4))
        style.map("TButton",
                  background=[("active", "#45475a"), ("pressed", "#585b70")])
        style.configure("Accent.TButton", background=ABTN, foreground=BG,
                        font=("Segoe UI", 10, "bold"), padding=(10, 5))
        style.map("Accent.TButton",
                  background=[("active", "#74c7ec"), ("pressed", "#89dceb")])
        style.configure("TScale",      background=BG)
        style.configure("TEntry",      fieldbackground="#313244",
                                       foreground=FG, insertcolor=FG)
        style.configure("TNotebook",   background=BG)
        style.configure("TNotebook.Tab", background="#313244", foreground=FG,
                        padding=(10, 4), font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", "#45475a")],
                  foreground=[("selected", ABTN)])

        # ---- layout principal ----------------------------------------------
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        left_col = ttk.Frame(main, width=310)
        left_col.pack(side="left", fill="y", padx=(0, 10))
        left_col.pack_propagate(False)

        right_col = ttk.Frame(main)
        right_col.pack(side="left", fill="both", expand=True)

        # ---- coluna esquerda -----------------------------------------------
        tk.Label(left_col, text="🌿 3D Root Analysis",
                 bg="#1e1e2e", fg="#a6e3a1",
                 font=("Segoe UI", 14, "bold")).pack(pady=(0, 8))

        # ZIPs
        frm_zip = ttk.LabelFrame(left_col, text="Arquivos de entrada")
        frm_zip.pack(fill="x", pady=3)
        self._make_file_row(frm_zip, "b0207.zip", self.zip_b0207, "b0207")
        self._make_file_row(frm_zip, "b0309.zip", self.zip_b0309, "b0309")

        # parâmetros
        frm_par = ttk.LabelFrame(left_col, text="Parâmetros de processamento")
        frm_par.pack(fill="x", pady=3)
        self._make_slider(frm_par, "Isovalor:",      self.iso_val,   0.10, 0.90, 0.05, fmt=".2f")
        self._make_slider(frm_par, "Suavização:",    self.smooth_it, 10,   200,  10,   fmt="d")
        self._make_slider(frm_par, "Decimação (%):", self.decim_t,   0.10, 0.95, 0.05, fmt=".2f")

        # botão processar
        ttk.Button(left_col, text="⚙  Carregar & Processar",
                   style="Accent.TButton",
                   command=self._start_processing).pack(fill="x", pady=5)

        ttk.Separator(left_col, orient="horizontal").pack(fill="x", pady=4)

        # visualizações individuais
        frm_vis = ttk.LabelFrame(left_col, text="Visualizações")
        frm_vis.pack(fill="x", pady=3)

        vis_buttons = [
            ("📦  DVR – b0207",              lambda: self._show_dvr("b0207")),
            ("📦  DVR – b0309",              lambda: self._show_dvr("b0309")),
            ("🌐  Isosuperfície – b0207",    lambda: self._show_iso("b0207")),
            ("🌐  Isosuperfície – b0309",    lambda: self._show_iso("b0309")),
            ("🦴  Iso + Esqueleto – b0207",  lambda: self._show_skeleton("b0207")),
            ("🦴  Iso + Esqueleto – b0309",  lambda: self._show_skeleton("b0309")),
            ("🔀  Janela Dividida – b0207",  lambda: self._show_split("b0207")),
            ("🔀  Janela Dividida – b0309",  lambda: self._show_split("b0309")),
        ]
        for label, cmd in vis_buttons:
            ttk.Button(frm_vis, text=label, command=cmd).pack(fill="x", pady=1)

        ttk.Separator(left_col, orient="horizontal").pack(fill="x", pady=4)
        ttk.Button(left_col, text="📊  Ver Métricas",
                   command=self._show_metrics_tab).pack(fill="x", pady=1)
        ttk.Button(left_col, text="📄  Gerar Relatório",
                   command=self._show_report).pack(fill="x", pady=1)

        # ---- coluna direita (notebook) -------------------------------------
        self.notebook = ttk.Notebook(right_col)
        self.notebook.pack(fill="both", expand=True)

        # aba log
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text=" 📋 Log ")
        self.log_box = scrolledtext.ScrolledText(
            log_frame, bg="#181825", fg="#cdd6f4",
            font=("Consolas", 9), state="disabled",
            relief="flat", borderwidth=0
        )
        self.log_box.pack(fill="both", expand=True)

        # aba métricas
        met_frame = ttk.Frame(self.notebook)
        self.notebook.add(met_frame, text=" 📊 Métricas ")
        self.met_text = scrolledtext.ScrolledText(
            met_frame, bg="#181825", fg="#a6e3a1",
            font=("Consolas", 10), state="disabled",
            relief="flat", borderwidth=0
        )
        self.met_text.pack(fill="both", expand=True)

        # aba relatório
        rep_frame = ttk.Frame(self.notebook)
        self.notebook.add(rep_frame, text=" 📝 Relatório ")
        self.rep_text = scrolledtext.ScrolledText(
            rep_frame, bg="#181825", fg="#cdd6f4",
            font=("Segoe UI", 10), state="disabled",
            relief="flat", borderwidth=0, wrap="word"
        )
        self.rep_text.pack(fill="both", expand=True)

        self._log("Bem-vindo! Selecione os ZIPs e clique em 'Carregar & Processar'.")
        self._log(f"Parâmetros padrão: isovalor={ISO_VALUE}, suavização={SMOOTH_ITER} "
                  f"iterações, decimação={DECIMATE_T*100:.0f}%.")

    # -----------------------------------------------------------------------
    def _make_file_row(self, parent, label, var, tag):
        frm = ttk.Frame(parent)
        frm.pack(fill="x", padx=4, pady=2)
        ttk.Label(frm, text=label, width=9).pack(side="left")
        ttk.Entry(frm, textvariable=var, width=15).pack(side="left", fill="x", expand=True)
        ttk.Button(frm, text="…",
                   command=lambda v=var, t=tag: self._browse(v, t),
                   width=3).pack(side="left")

    def _make_slider(self, parent, label, var, from_, to, resolution, fmt="f"):
        frm = ttk.Frame(parent)
        frm.pack(fill="x", padx=4, pady=2)
        ttk.Label(frm, text=label, width=15).pack(side="left")
        sc = ttk.Scale(frm, variable=var, from_=from_, to=to, orient="horizontal")
        sc.pack(side="left", fill="x", expand=True)
        # Label dinâmico que mostra o valor atual formatado
        val_lbl = ttk.Label(frm, width=6)
        val_lbl.pack(side="left")

        def _update_label(*_):
            v = var.get()
            if fmt == "d":
                val_lbl.config(text=f"{int(v)}")
            else:
                val_lbl.config(text=f"{v:{fmt}}")
        var.trace_add("write", _update_label)
        _update_label()

    # -----------------------------------------------------------------------
    def _browse(self, var: tk.StringVar, tag: str):
        path = filedialog.askopenfilename(
            title=f"Selecionar {tag}.zip",
            filetypes=[("ZIP files", "*.zip"), ("All", "*.*")]
        )
        if path:
            var.set(path)

    # -----------------------------------------------------------------------
    def _log(self, msg: str):
        """Escreve no log de forma thread-safe."""
        def _do():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(0, _do)

    # -----------------------------------------------------------------------
    def _start_processing(self):
        p207 = self.zip_b0207.get().strip()
        p309 = self.zip_b0309.get().strip()

        if not p207 or not p309:
            messagebox.showerror("Erro", "Selecione ambos os ZIPs antes de processar.")
            return
        for path in [p207, p309]:
            if not os.path.isfile(path):
                messagebox.showerror("Erro", f"Arquivo não encontrado:\n{path}")
                return

        iso_v  = self.iso_val.get()
        sm_it  = int(self.smooth_it.get())
        dec_t  = self.decim_t.get()

        self._log("=" * 55)
        self._log("Iniciando processamento das duas raízes...")
        self._log(f"  Isovalor={iso_v:.2f}  Suavização={sm_it}  Decimação={dec_t*100:.0f}%")
        self._log("=" * 55)

        def worker():
            for path, tag in [(p207, "b0207"), (p309, "b0309")]:
                try:
                    rd = RootData(tag, path, log_fn=self._log)
                    rd.run_full_pipeline(
                        iso_val        = iso_v,
                        smooth_iter    = sm_it,
                        decimate_target= dec_t
                    )
                    self.roots[tag] = rd
                except Exception as exc:
                    self._log(f"[{tag}] ERRO: {exc}")
                    self.after(0, lambda e=exc, t=tag:
                               messagebox.showerror("Erro",
                                                    f"Falha ao processar {t}:\n{e}"))
                    return
            self.after(0, self._update_metrics_tab)
            self._log("")
            self._log("✅ Processamento concluído!")
            self._log("Use os botões à esquerda para visualizar e gerar relatório.")

        threading.Thread(target=worker, daemon=True).start()

    # -----------------------------------------------------------------------
    def _require_root(self, tag: str):
        if tag not in self.roots:
            messagebox.showwarning("Aviso", f"Raiz '{tag}' ainda não processada.\n"
                                   "Clique em 'Carregar & Processar' primeiro.")
            return None
        return self.roots[tag]

    # -----------------------------------------------------------------------
    # ---- (3a) DVR – Direct Volume Rendering --------------------------------
    def _show_dvr(self, tag: str):
        rd = self._require_root(tag)
        if rd is None:
            return

        def _render():
            plotter = pv.Plotter(title=f"DVR – {tag}")
            plotter.set_background("black")

            # Tabela de transferência: fundo transparente, raízes opacas
            opacity  = [0.0, 0.0, 0.05, 0.2, 0.8, 1.0]
            plotter.add_volume(
                rd.grid, scalars="scalars",
                opacity=opacity,
                cmap="bone",
                shade=True,
                diffuse=0.6, specular=0.3, ambient=0.3
            )
            plotter.add_text(f"DVR – {tag}", font_size=12, color="white",
                             position="upper_left")
            plotter.show()

        threading.Thread(target=_render, daemon=True).start()

    # -----------------------------------------------------------------------
    # ---- (3b) Isosuperfície ------------------------------------------------
    def _show_iso(self, tag: str):
        rd = self._require_root(tag)
        if rd is None:
            return

        def _render():
            plotter = pv.Plotter(title=f"Isosuperfície – {tag}")
            plotter.set_background("black")
            plotter.add_mesh(rd.iso_decim,
                             color="#c8a96e",
                             smooth_shading=True,
                             specular=0.3)
            plotter.add_text(f"Isosuperfície (suavizada + decimada) – {tag}",
                             font_size=11, color="white", position="upper_left")
            plotter.show()

        threading.Thread(target=_render, daemon=True).start()

    # -----------------------------------------------------------------------
    # ---- (3c) Isosuperfície + Esqueleto ------------------------------------
    def _show_skeleton(self, tag: str):
        rd = self._require_root(tag)
        if rd is None:
            return

        def _render():
            plotter = pv.Plotter(title=f"Isosuperfície + Esqueleto – {tag}")
            plotter.set_background("black")

            # Isosuperfície semi-transparente
            plotter.add_mesh(rd.iso_decim,
                             color="#c8a96e", opacity=0.30,
                             smooth_shading=True)

            # Esqueleto: linhas azul-claro
            if rd.skel_lines is not None and rd.skel_lines.n_points > 0:
                if rd.skel_lines.n_cells > 0:
                    plotter.add_mesh(rd.skel_lines,
                                     color="#89b4fa",
                                     line_width=2,
                                     render_lines_as_tubes=True)
                else:
                    # Fallback: apenas pontos
                    plotter.add_points(rd.skel_lines.points,
                                       color="#89b4fa",
                                       point_size=2,
                                       render_points_as_spheres=True)

            plotter.add_text(f"Isosuperfície + Esqueleto – {tag}",
                             font_size=11, color="white", position="upper_left")
            plotter.show()

        threading.Thread(target=_render, daemon=True).start()

    # -----------------------------------------------------------------------
    # ---- (3e) Janela dividida e sincronizada (DVR | Isosuperfície) ---------
    def _show_split(self, tag: str):
        rd = self._require_root(tag)
        if rd is None:
            return

        def _render():
            plotter = pv.Plotter(
                shape=(1, 2),
                title=f"Janela Dividida Sincronizada – {tag}"
            )
            plotter.set_background("black")

            # Painel esquerdo: DVR
            plotter.subplot(0, 0)
            opacity = [0.0, 0.0, 0.05, 0.2, 0.8, 1.0]
            plotter.add_volume(rd.grid, scalars="scalars",
                               opacity=opacity, cmap="bone", shade=True)
            plotter.add_text("DVR", font_size=14, color="white",
                             position="upper_left")

            # Painel direito: Isosuperfície
            plotter.subplot(0, 1)
            plotter.add_mesh(rd.iso_decim, color="#c8a96e",
                             smooth_shading=True, specular=0.3)
            plotter.add_text("Isosuperfície", font_size=14, color="white",
                             position="upper_left")

            # Câmeras sincronizadas
            plotter.link_views()
            plotter.show()

        threading.Thread(target=_render, daemon=True).start()

    # -----------------------------------------------------------------------
    # ---- (3d) Aba de métricas ----------------------------------------------
    def _show_metrics_tab(self):
        if not self.roots:
            messagebox.showwarning("Aviso", "Processe pelo menos uma raiz primeiro.")
            return
        self._update_metrics_tab()

    def _update_metrics_tab(self):
        self.met_text.configure(state="normal")
        self.met_text.delete("1.0", "end")
        W = 58

        for tag in ["b0207", "b0309"]:
            if tag not in self.roots:
                continue
            rd = self.roots[tag]
            m  = rd.metrics
            z, y, x = rd.volume_clean.shape

            self.met_text.insert("end", f"\n{'═'*W}\n")
            self.met_text.insert("end", f"  RAIZ: {tag.upper()}\n")
            self.met_text.insert("end", f"{'═'*W}\n")
            self.met_text.insert("end", f"  Arquivo   : {os.path.basename(rd.zip_path)}\n")
            self.met_text.insert("end", f"  Dimensões : {x} × {y} × {z} voxels\n")

            self.met_text.insert("end", f"\n  {'─'*W}\n  MÉTRICAS GEOMÉTRICAS\n  {'─'*W}\n")
            self.met_text.insert("end", f"  Volume (voxels binários) : {m['volume_voxels']:>12.0f}\n")
            self.met_text.insert("end", f"  Área de superfície       : {m['area_superficie']:>12.2f}  unid²\n")
            self.met_text.insert("end", f"  Compacidade              : {m['compacidade']:>12.6f}  (1=esfera)\n")
            self.met_text.insert("end", f"  Excentricidade           : {m['excentricidade']:>12.4f}  (0=esfera)\n")

            self.met_text.insert("end", f"\n  {'─'*W}\n  MÉTRICAS DO ESQUELETO (7 métricas)\n  {'─'*W}\n")
            self.met_text.insert("end", f"  1. Voxels totais         : {m['skel_total_voxels']:>12d}\n")
            self.met_text.insert("end", f"  2. Pontos de ramificação : {m['skel_branch_points']:>12d}\n")
            self.met_text.insert("end", f"  3. Pontos terminais      : {m['skel_tip_points']:>12d}\n")
            self.met_text.insert("end", f"  4. Densidade             : {m['skel_density']:>12.4f}\n")
            self.met_text.insert("end", f"  5. Alongamento (BB)      : {m['skel_elongation']:>12.2f}\n")
            self.met_text.insert("end", f"  6. Tortuosidade          : {m['skel_tortuosity']:>12.2f}\n")
            self.met_text.insert("end", f"  7. Nº de segmentos       : {m['skel_segments']:>12d}\n")

        # Comparação lado a lado (se ambas disponíveis)
        if "b0207" in self.roots and "b0309" in self.roots:
            self.met_text.insert("end", f"\n\n{'═'*W}\n")
            self.met_text.insert("end", f"  COMPARAÇÃO: b0207  ×  b0309\n")
            self.met_text.insert("end", f"{'═'*W}\n")
            m207 = self.roots["b0207"].metrics
            m309 = self.roots["b0309"].metrics
            fmt = "  {:<26s}  {:>12}  {:>12}\n"
            self.met_text.insert("end", fmt.format("Métrica", "b0207", "b0309"))
            self.met_text.insert("end", f"  {'─'*52}\n")
            rows = [
                ("Volume (voxels)",       f"{m207['volume_voxels']:.0f}",        f"{m309['volume_voxels']:.0f}"),
                ("Área superfície",       f"{m207['area_superficie']:.2f}",       f"{m309['area_superficie']:.2f}"),
                ("Compacidade",           f"{m207['compacidade']:.6f}",           f"{m309['compacidade']:.6f}"),
                ("Excentricidade",        f"{m207['excentricidade']:.4f}",        f"{m309['excentricidade']:.4f}"),
                ("Skel – voxels",         f"{m207['skel_total_voxels']}",         f"{m309['skel_total_voxels']}"),
                ("Skel – ramificações",   f"{m207['skel_branch_points']}",        f"{m309['skel_branch_points']}"),
                ("Skel – pontas",         f"{m207['skel_tip_points']}",           f"{m309['skel_tip_points']}"),
                ("Skel – densidade",      f"{m207['skel_density']:.4f}",          f"{m309['skel_density']:.4f}"),
                ("Skel – alongamento",    f"{m207['skel_elongation']:.2f}",       f"{m309['skel_elongation']:.2f}"),
                ("Skel – tortuosidade",   f"{m207['skel_tortuosity']:.2f}",       f"{m309['skel_tortuosity']:.2f}"),
                ("Skel – segmentos",      f"{m207['skel_segments']}",             f"{m309['skel_segments']}"),
            ]
            for r in rows:
                self.met_text.insert("end", fmt.format(*r))

        self.met_text.configure(state="disabled")
        self.notebook.select(1)

    # -----------------------------------------------------------------------
    # ---- (3f) Relatório ----------------------------------------------------
    def _show_report(self):
        if len(self.roots) < 2:
            messagebox.showwarning("Aviso", "Processe ambas as raízes antes de gerar o relatório.")
            return

        r207 = self.roots["b0207"]
        r309 = self.roots["b0309"]
        m207 = r207.metrics
        m309 = r309.metrics
        z207, y207, x207 = r207.volume_clean.shape
        z309, y309, x309 = r309.volume_clean.shape

        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║       RELATÓRIO – 3º TRABALHO PRÁTICO DE VISÃO COMPUTACIONAL        ║
║       UNICENTRO / DECOMP – Prof. Dr. Mauro Miazaki                  ║
╚══════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PRÉ-PROCESSAMENTO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Carregamento:
  As imagens foram carregadas diretamente dos arquivos ZIP (b0207.zip e
  b0309.zip), sem extração em disco. Cada arquivo contém uma sequência de
  fatias PNG em escala de cinza representando cortes tomográficos das raízes.
  As fatias foram ordenadas lexicograficamente e empilhadas em um volume
  tridimensional NumPy de formato (Z, Y, X).

Dimensões dos volumes:
  • b0207: {x207} × {y207} × {z207} voxels  ({x207*y207*z207/1_000_000:.2f} Mvoxels)
  • b0309: {x309} × {y309} × {z309} voxels  ({x309*y309*z309/1_000_000:.2f} Mvoxels)

Limpeza de bordas (border cleaning):
  Os {BORDER_CLEAN} pixels das bordas externas nas direções X e Y, bem como as
  {BORDER_CLEAN} primeiras e últimas fatias na direção Z, foram zerados.
  Essa operação elimina artefatos típicos de escâneres (reflexões de borda,
  ruído de truncamento e gradientes artificiais) que, caso mantidos, seriam
  incluídos erroneamente na isosuperfície e no esqueleto, comprometendo as
  métricas. A limpeza é realizada antes de qualquer normalização.

  Após a limpeza, os volumes foram normalizados para o intervalo [0, 1]
  (divisão pelo valor máximo) e convertidos para o formato pv.ImageData
  do PyVista, com os escalares armazenados como ponto flutuante de 32 bits.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. VISUALIZAÇÕES GERADAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

a) DVR – Direct Volume Rendering:
   Renderização volumétrica direta usando a função add_volume() do PyVista.
   Uma tabela de transferência de opacidade torna o fundo (intensidades
   baixas) completamente transparente e as regiões densas da raiz com alta
   opacidade. O colormap "bone" realça a estrutura radicular em tons claros
   sobre o fundo preto. Parâmetros de iluminação: diffuse=0.6, specular=0.3,
   ambient=0.3.

b) Isosuperfície (suavização e decimação):
   • Geração: Marching Cubes via pyvista .contour() com isovalor = {self.iso_val.get():.2f}.
   • Suavização: filtro Laplaciano ({int(self.smooth_it.get())} iterações, fator de
     relaxação 0.1, suavização de borda ativada). Remove o efeito de
     "escada" dos voxels.
   • Decimação: redução de {self.decim_t.get()*100:.0f}% das faces via pyvista .decimate(),
     mantendo a forma geral da raiz com menor custo computacional.

c) Esqueleto 3D + visualização combinada:
   • Binarização: limiarização de Otsu aplicada apenas nos pixels não-nulos
     para evitar viés do fundo preto.
   • Limpeza: remoção de objetos com menos de 64 voxels e fechamento
     morfológico com bola de raio 1 (preenchimento de pequenos buracos).
   • Skeletonize: skeletonize_3d() do scikit-image (algoritmo de afinamento
     topológico 3D, preserva a conectividade).
   • Conversão para linhas: os voxels do esqueleto são conectados com seus
     vizinhos 26-conectados para gerar células de linha em pv.PolyData.
   • Visualização: isosuperfície semi-transparente (opacidade 0.30) com o
     esqueleto sobreposto em azul como tubos de linha.

d) Métricas: ver seção 3.

e) Janela dividida e sincronizada:
   Duas subjanelas lado a lado usando pv.Plotter(shape=(1,2)):
   • Esquerda: DVR com a mesma tabela de transferência de opacidade.
   • Direita: Isosuperfície suavizada e decimada.
   As câmeras são sincronizadas via plotter.link_views(): qualquer rotação,
   zoom ou translação aplicados em uma subjanela refletem imediatamente
   na outra.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. MÉTRICAS OBTIDAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                              b0207              b0309
  ────────────────────────────────────────────────────────────────
  Volume (voxels)    {m207['volume_voxels']:>16.0f}   {m309['volume_voxels']:>16.0f}
  Área superfície    {m207['area_superficie']:>16.2f}   {m309['area_superficie']:>16.2f}
  Compacidade        {m207['compacidade']:>16.6f}   {m309['compacidade']:>16.6f}
  Excentricidade     {m207['excentricidade']:>16.4f}   {m309['excentricidade']:>16.4f}
  ────────────────────────────────────────────────────────────────
  Skel – voxels      {m207['skel_total_voxels']:>16d}   {m309['skel_total_voxels']:>16d}
  Skel – ramos       {m207['skel_branch_points']:>16d}   {m309['skel_branch_points']:>16d}
  Skel – pontas      {m207['skel_tip_points']:>16d}   {m309['skel_tip_points']:>16d}
  Skel – densidade   {m207['skel_density']:>16.4f}   {m309['skel_density']:>16.4f}
  Skel – alongament. {m207['skel_elongation']:>16.2f}   {m309['skel_elongation']:>16.2f}
  Skel – tortuosid.  {m207['skel_tortuosity']:>16.2f}   {m309['skel_tortuosity']:>16.2f}
  Skel – segmentos   {m207['skel_segments']:>16d}   {m309['skel_segments']:>16d}

Definições das métricas:
  • Volume: número de voxels classificados como raiz após binarização Otsu.
  • Área de superfície: soma das áreas das faces triangulares da malha
    decimada (mesh.area do PyVista).
  • Compacidade: (36π·V²)/A³ – vale 1 para uma esfera perfeita; quanto
    menor, mais irregular é a forma.
  • Excentricidade: calculada via autovalores da matriz de covariância dos
    voxels. Varia de 0 (esfera) a 1 (bastão).
  • Skel – voxels: comprimento total do esqueleto em voxels.
  • Skel – ramos: voxels com ≥ 3 vizinhos no esqueleto (pontos de
    bifurcação), indicam complexidade da ramificação.
  • Skel – pontas: voxels com exatamente 1 vizinho, indicam extremidades.
  • Skel – densidade: razão voxels_esqueleto / voxels_binários.
  • Skel – alongamento: dimensão_máxima / dimensão_mínima do bounding box.
  • Skel – tortuosidade: comprimento_skel / diagonal_bounding_box. Valores
    maiores indicam trajetórias mais sinuosas.
  • Skel – segmentos: número de componentes conexos após remover os pontos
    de ramificação (estimativa do número de ramos individuais).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. COMPARAÇÃO E ANÁLISE DAS DUAS RAÍZES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{self._generate_comparison(m207, m309)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. METODOLOGIA TÉCNICA RESUMIDA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Linguagem    : Python 3
  Interface    : tkinter (GUI nativa, tema "clam" customizado)
  Visualização : PyVista (renderização 3D interativa, VTK backend)
  Processamento: scikit-image (Otsu, morfologia, skeletonize_3d),
                 scipy.ndimage (convolução, label), NumPy, Pillow

  Pipeline completo:
    ZIP ──► PIL (grayscale) ──► ndarray (Z,Y,X) uint8
        ──► limpeza de bordas (border={BORDER_CLEAN}px)
        ──► normalização [0,1] ──► pv.ImageData
        ──► isosuperfície (Marching Cubes)
        ──► suavização Laplaciana ──► decimação
        ──► binarização Otsu ──► limpeza morfológica
        ──► skeletonize_3d ──► linhas PyVista
        ──► cálculo de métricas
"""

        self.rep_text.configure(state="normal")
        self.rep_text.delete("1.0", "end")
        self.rep_text.insert("end", report)
        self.rep_text.configure(state="disabled")
        self.notebook.select(2)

        # Oferecer salvar o relatório
        if messagebox.askyesno("Salvar relatório",
                               "Deseja salvar o relatório em arquivo .txt?"):
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Texto", "*.txt"), ("Todos", "*.*")],
                initialfile="relatorio_raizes.txt"
            )
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(report)
                self._log(f"Relatório salvo em: {path}")

    # -----------------------------------------------------------------------
    def _generate_comparison(self, m207: dict, m309: dict) -> str:
        """Gera texto de análise comparativa com base nas métricas."""
        lines = []

        def _cmp_line(label, key, fmt=".2f", unit=""):
            v207 = m207[key]
            v309 = m309[key]
            if v309 > 0:
                pct = abs(v207 / v309 - 1) * 100
            else:
                pct = 0.0
            maior = "b0207" if v207 > v309 else ("b0309" if v309 > v207 else "igual")
            lines.append(
                f"  • {label:<26s}: b0207={v207:{fmt}}{unit}  "
                f"b0309={v309:{fmt}}{unit}  → {maior} é {pct:.1f}% maior"
            )

        _cmp_line("Volume (voxels)",      "volume_voxels",     ".0f", " vox")
        _cmp_line("Área superfície",      "area_superficie",   ".2f", " unid²")
        _cmp_line("Compacidade",          "compacidade",       ".6f")
        _cmp_line("Excentricidade",       "excentricidade",    ".4f")
        _cmp_line("Skel – voxels",        "skel_total_voxels", ".0f")
        _cmp_line("Skel – ramos",         "skel_branch_points",".0f")
        _cmp_line("Skel – pontas",        "skel_tip_points",   ".0f")
        _cmp_line("Skel – densidade",     "skel_density",      ".4f")
        _cmp_line("Skel – alongamento",   "skel_elongation",   ".2f")
        _cmp_line("Skel – tortuosidade",  "skel_tortuosity",   ".2f")
        _cmp_line("Skel – segmentos",     "skel_segments",     ".0f")

        lines.append("")
        lines.append("  ── Interpretação qualitativa ──")

        # Volume
        vr = m207["volume_voxels"] / max(m309["volume_voxels"], 1)
        if vr > 1.20:
            lines.append("  → Volume: b0207 possui volume consideravelmente maior, indicando")
            lines.append("    uma raiz com mais massa de tecido ou diâmetro médio superior.")
        elif vr < 0.80:
            lines.append("  → Volume: b0309 possui volume consideravelmente maior.")
        else:
            lines.append("  → Volume: as duas raízes apresentam volumes comparáveis.")

        # Ramificação
        br207 = m207["skel_branch_points"]
        br309 = m309["skel_branch_points"]
        if br207 > br309 * 1.20:
            lines.append("  → Ramificação: b0207 tem mais pontos de bifurcação, sugerindo")
            lines.append("    arquitetura radicular mais complexa e ramificada.")
        elif br309 > br207 * 1.20:
            lines.append("  → Ramificação: b0309 tem mais pontos de bifurcação (maior")
            lines.append("    complexidade topológica).")
        else:
            lines.append("  → Ramificação: complexidade de bifurcação similar entre as raízes.")

        # Alongamento
        if m207["skel_elongation"] > m309["skel_elongation"]:
            lines.append("  → Forma: b0207 é mais alongada (estrutura mais linear/vertical).")
        elif m309["skel_elongation"] > m207["skel_elongation"]:
            lines.append("  → Forma: b0309 é mais alongada (estrutura mais linear/vertical).")

        # Tortuosidade
        if m207["skel_tortuosity"] > m309["skel_tortuosity"]:
            lines.append("  → Tortuosidade: b0207 possui trajetórias mais sinuosas.")
        else:
            lines.append("  → Tortuosidade: b0309 possui trajetórias mais sinuosas.")

        # Compacidade
        if m207["compacidade"] > m309["compacidade"]:
            lines.append("  → Compacidade: b0207 é mais compacta (morfologia mais globosa).")
        else:
            lines.append("  → Compacidade: b0309 é mais compacta (morfologia mais globosa).")

        # Excentricidade
        if m207["excentricidade"] > m309["excentricidade"]:
            lines.append("  → Excentricidade: b0207 tem distribuição de massa mais assimétrica.")
        else:
            lines.append("  → Excentricidade: b0309 tem distribuição de massa mais assimétrica.")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Garante que o splitting da janela dividida fique equilibrado
    pv.global_theme.multi_rendering_splitting_position = 0.5

    app = App()
    app.mainloop()