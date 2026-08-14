
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import mujoco
import mujoco.viewer
import os
import time


def _list_npz_files(root_dir: str) -> dict[str, list[str]]:
    """
    Return {relative_folder: [absolute_npz_paths...]} for all .npz under root_dir (recursive).
    """
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    folder2files: dict[str, list[str]] = {}
    if not os.path.isdir(root_dir):
        return folder2files

    for dirpath, _, filenames in os.walk(root_dir):
        npzs = [f for f in filenames if f.lower().endswith(".npz")]
        if not npzs:
            continue
        rel = os.path.relpath(dirpath, root_dir)
        rel = "." if rel == "." else rel.replace("\\", "/")
        folder2files.setdefault(rel, [])
        for f in sorted(npzs):
            folder2files[rel].append(os.path.join(dirpath, f))
    return folder2files


class NPZAnalyzer:
    def __init__(self, npz_path: str | None = None, *, npz_root_dir: str | None = None):
        self.camera_follow = False
        self.tags: dict[int, list[str]] = {}
        self.plot_windows = []

        # --- playlist / selection state ---
        self.npz_root_dir = os.path.abspath(os.path.expanduser(npz_root_dir)) if npz_root_dir else ""
        self.folder2files: dict[str, list[str]] = {}
        self.selected_folder = "."
        self.checked_paths: set[str] = set()  # abs npz paths checked in tree
        self.npz_item2path: dict[str, str] = {}  # tree item id -> abs npz path (files only)
        self.play_queue: list[str] = []
        self.play_queue_idx = 0
        self.current_npz_path: str | None = None

        # --- mujoco init (shared across files) ---
        self.model = mujoco.MjModel.from_xml_path(
            "assets/unitree_g1/scene_mjx.xml"
        )
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # --- UI init ---
        self.root = tk.Tk()
        self.root.title("NPZ 超限检测工具")
        self.root.geometry("1600x800")

        self.current_frame = 0
        self.is_playing = False

        # Placeholders: will be set by load_npz()
        self.npzdata = None
        self.file_name = ""
        self.qpos_23d = None
        self.root_msg = None
        self.foot_contact = None
        self.contact_modified = False
        self.modified_foot_contact = None
        self.num_frames = 0
        self.num_joints = 0
        self.joint_mode = None
        self.dt = 1 / 50
        self.velocities = None
        self.qpos_29d = None

        # joint UI holders
        self.joint_names = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.velocity_limits = []

        self.value_labels = []
        self.velocity_labels = []
        self.limit_labels = []
        self.velocity_limit_labels = []
        self.status_labels = []
        self.velocity_status_labels = []

        self.analysis_frame = None
        self.scrollable_frame = None
        self.frame_slider = None
        self.play_button = None
        self.frame_var = None
        self.frame_entry_var = None
        self.current_tag_text = None
        self.current_tags_var = None
        self.camera_follow_var = None
        self.left_contact_var = None
        self.right_contact_var = None

        # selection widgets
        self.root_dir_var = tk.StringVar(value=self.npz_root_dir)
        self.folder_var = tk.StringVar(value=".")
        self.npz_tree = None

        self.setup_ui()

        # initial scan + initial file load
        if self.npz_root_dir:
            self.refresh_npz_tree()

        if npz_path:
            self.load_npz(npz_path)
        else:
            # If there's a root dir and at least one npz, load the first one.
            first = self._get_first_npz_in_root()
            if first:
                self.load_npz(first)

    def _get_first_npz_in_root(self) -> str | None:
        if not self.npz_root_dir:
            return None
        folder2files = _list_npz_files(self.npz_root_dir)
        for _, files in sorted(folder2files.items(), key=lambda x: x[0]):
            if files:
                return files[0]
        return None

    # --------------------- NPZ loading / state ---------------------
    def _infer_joint_mode(self, qpos: np.ndarray) -> list[str]:
        num_joints = qpos.shape[1]
        if (num_joints - 7) == 23:
            return ['66155']
        if (num_joints - 7) == 29:
            return ['66377']
        raise ValueError(f"Invalid joint numbers: qpos has shape {qpos.shape}")

    def _set_joint_limits_by_mode(self, joint_mode: list[str]):
        if joint_mode == ['66155']:
            self.joint_names = [
                "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
                "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
                "waist_yawt", "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
                "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll"
            ]

            self.lower_bounds = [
                -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
                -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
                -2.618, -3.0892, -1.5882, -2.618, -1.0472, -1.97222,
                -3.0892, -2.2515, -2.618, -1.0472, -1.97222
            ]

            self.upper_bounds = [
                2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
                2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
                2.618, 2.6704, 2.2515, 2.618, 2.0944, 1.97222,
                2.6704, 1.5882, 2.618, 2.0944, 1.97222
            ]

            self.velocity_limits = [
                32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
                32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
                32.0, 37.0, 37.0, 37.0, 37.0, 37.0,
                37.0, 37.0, 37.0, 37.0, 37.0
            ]
        elif joint_mode == ['66377']:
            self.joint_names = [
                "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
                "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
                "waist_yaw", "waist_roll", "waist_pitch",
                "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
                "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"
            ]

            self.lower_bounds = [
                -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
                -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
                -2.618, -0.52, -0.52,
                -3.0892, -1.5882, -2.618, -1.0472, -1.97222, -1.61443, -1.61443,
                -3.0892, -2.2515, -2.618, -1.0472, -1.97222, -1.61443, -1.61443
            ]

            self.upper_bounds = [
                2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
                2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
                2.618, 0.52, 0.52,
                2.6704, 2.2515, 2.618, 2.0944, 1.97222, 1.61443, 1.61443,
                2.6704, 1.5882, 2.618, 2.0944, 1.97222, 1.61443, 1.61443
            ]

            self.velocity_limits = [
                32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
                32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
                32.0, 37.0, 37.0,
                37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0,
                37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0
            ]
        else:
            raise ValueError(f"Unknown joint mode: {joint_mode}")

    def load_npz(self, npz_path: str):
        npz_path = os.path.abspath(os.path.expanduser(npz_path))
        try:
            npzdata = np.load(npz_path, allow_pickle=True)
        except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError) as e:
            messagebox.showerror("加载失败", f"无法打开NPZ:\n{npz_path}\n\n{e}")
            return

        if "qpos" not in npzdata:
            messagebox.showerror("格式错误", f"NPZ中缺少qpos:\n{npz_path}")
            return

        self.npzdata = npzdata
        self.current_npz_path = npz_path
        self.file_name = os.path.basename(npz_path)
        self.root.title(f"NPZ 超限检测工具 - {self.file_name}")

        self.tags = {}
        self.contact_modified = False

        self.qpos_23d = self.npzdata["qpos"].copy()
        self.root_msg = self.npzdata["qpos"][:, :7].copy()

        self.foot_contact = None
        if "foot_contact" in self.npzdata:
            self.foot_contact = self.npzdata["foot_contact"]
        self.modified_foot_contact = self.foot_contact.copy() if self.foot_contact is not None else None

        self.num_frames, self.num_joints = self.qpos_23d.shape

        # infer mode + limits (and rebuild joint UI if needed)
        new_mode = self._infer_joint_mode(self.qpos_23d)
        mode_changed = (self.joint_mode != new_mode)
        self.joint_mode = new_mode
        self._set_joint_limits_by_mode(self.joint_mode)

        # dt + velocities
        self.dt = 1 / 50
        self.velocities = np.zeros_like(self.qpos_23d)
        self.velocities[1:] = (self.qpos_23d[1:] - self.qpos_23d[:-1]) / self.dt
        self.velocities[0] = 0

        # 29d qpos helper
        self.qpos_29d = np.zeros((self.num_frames, 29))
        self.qpos_29d[:, :7] = self.qpos_23d[:, :7]
        if self.joint_mode == ['66155']:
            self.qpos_29d[:, [0, 1, 2, 3, 4, 5,
                              6, 7, 8, 9, 10, 11,
                              12,
                              15, 16, 17, 18, 19,
                              22, 23, 24, 25, 26]] = self.qpos_23d[:, 7:]
        elif self.joint_mode == ['66377']:
            self.qpos_29d[:, [0, 1, 2, 3, 4, 5,
                              6, 7, 8, 9, 10, 11,
                              12, 13, 14,
                              15, 16, 17, 18, 19, 20, 21,
                              22, 23, 24, 25, 26, 27, 28, ]] = self.qpos_23d[:, 7:]

        if mode_changed:
            self._rebuild_joint_analysis_ui()

        # slider bounds + labels
        if self.frame_slider is not None:
            self.frame_slider.configure(to=max(0, self.num_frames - 1))
        if self.frame_var is not None:
            self.current_frame = 0
            self.frame_var.set(f"1/{self.num_frames}")
        if self.frame_entry_var is not None:
            self.frame_entry_var.set("")

        self.load_frame(0)

    # --------------------- UI setup ---------------------
    def setup_ui(self):
        self.current_tag_text = tk.StringVar(value="")
        self.current_tags_var = tk.StringVar(value="当前帧Tag: 无")
        self.frame_var = tk.StringVar(value="0/0")
        self.camera_follow_var = tk.BooleanVar(value=self.camera_follow)
        self.frame_entry_var = tk.StringVar()
        self.left_contact_var = tk.StringVar()
        self.right_contact_var = tk.StringVar()

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.LabelFrame(main_frame, text="帧控制")
        control_frame.pack(fill=tk.X, pady=10)

        # --- LEFT COLUMN: folder + npz checklist ---
        select_col = ttk.Frame(control_frame)
        select_col.pack(side=tk.LEFT, padx=(5, 15), pady=5)

        dir_row = ttk.Frame(select_col)
        dir_row.pack(fill=tk.X)
        ttk.Label(dir_row, text="NPZ根目录:").pack(side=tk.LEFT, padx=(0, 5))
        dir_entry = ttk.Entry(dir_row, textvariable=self.root_dir_var, width=45)
        dir_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(dir_row, text="浏览", command=self.browse_root_dir).pack(side=tk.LEFT)
        ttk.Button(dir_row, text="刷新", command=self.refresh_npz_tree).pack(side=tk.LEFT, padx=(5, 0))

        # 折叠目录树（避免为每个npz创建Checkbutton导致资源耗尽）
        tree_frame = ttk.LabelFrame(select_col, text="NPZ选择（目录可折叠，点击文件切换勾选）")
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        tree_toolbar = ttk.Frame(tree_frame)
        tree_toolbar.pack(fill=tk.X, padx=4, pady=(4, 0))

        ttk.Button(tree_toolbar, text="全选(展开分支)", command=self.tree_select_all_expanded).pack(side=tk.LEFT)
        ttk.Button(tree_toolbar, text="全不选", command=self.tree_clear_all).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(tree_toolbar, text="折叠全部", command=self.tree_collapse_all).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(tree_toolbar, text="展开全部(谨慎)", command=self.tree_expand_all).pack(side=tk.LEFT, padx=(5, 0))

        self.npz_tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse", height=12)
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.npz_tree.yview)
        self.npz_tree.configure(yscrollcommand=tree_scroll.set)
        self.npz_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=4)

        self.npz_tree.bind("<<TreeviewOpen>>", self.on_tree_open)
        self.npz_tree.bind("<Button-1>", self.on_tree_click)

        # --- existing controls (right side) ---
        right_controls = ttk.Frame(control_frame)
        right_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)

        mode_label = ttk.Label(
            right_controls, text="Joint Mode: -", font=("TkDefaultFont", 10, "bold")
        )
        mode_label.pack(side=tk.LEFT, padx=10)
        self.mode_label = mode_label

        self.camera_follow_cb = ttk.Checkbutton(
            right_controls, text="相机跟随", variable=self.camera_follow_var, command=self.on_camera_follow_change
        )
        self.camera_follow_cb.pack(side=tk.LEFT, padx=10)

        frame_label = ttk.Label(right_controls, text="当前帧:")
        frame_label.pack(side=tk.LEFT, padx=5)

        frame_display = ttk.Label(right_controls, textvariable=self.frame_var)
        frame_display.pack(side=tk.LEFT, padx=5)

        prev_btn = ttk.Button(right_controls, text="上一帧", command=self.prev_frame)
        prev_btn.pack(side=tk.LEFT, padx=5)

        next_btn = ttk.Button(right_controls, text="下一帧", command=self.next_frame)
        next_btn.pack(side=tk.LEFT, padx=5)

        self.play_button = ttk.Button(right_controls, text="播放", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)

        plot_btn = ttk.Button(right_controls, text="绘制关节轨迹", command=self.plot_joint_trajectories)
        plot_btn.pack(side=tk.LEFT, padx=5)

        self.frame_slider = ttk.Scale(
            right_controls, from_=0, to=0, orient=tk.HORIZONTAL, length=300, command=self.on_frame_slider
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        frame_entry_frame = ttk.Frame(right_controls)
        frame_entry_frame.pack(side=tk.LEFT, padx=5)

        frame_entry = ttk.Entry(frame_entry_frame, textvariable=self.frame_entry_var, width=8)
        frame_entry.pack(side=tk.LEFT)

        frame_set_btn = ttk.Button(frame_entry_frame, text="跳转", width=4, command=self.go_to_frame)
        frame_set_btn.pack(side=tk.LEFT, padx=(5, 0))

        tag_frame = ttk.Frame(right_controls)
        tag_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(tag_frame, text="Tag:").pack(side=tk.LEFT, padx=(0, 2))
        self.tag_entry = ttk.Entry(tag_frame, textvariable=self.current_tag_text, width=15)
        self.tag_entry.pack(side=tk.LEFT, padx=2)
        self.tag_entry.bind("<Return>", self.add_tag)

        add_tag_btn = ttk.Button(tag_frame, text="添加Tag", command=self.add_tag)
        add_tag_btn.pack(side=tk.LEFT, padx=5)

        self.tags_label = ttk.Label(tag_frame, textvariable=self.current_tags_var)
        self.tags_label.pack(side=tk.LEFT, padx=5)

        view_tags_btn = ttk.Button(tag_frame, text="查看所有Tag", command=self.view_all_tags)
        view_tags_btn.pack(side=tk.LEFT, padx=5)

        # analysis area
        self.analysis_frame = ttk.LabelFrame(main_frame, text="关节分析")
        self.analysis_frame.pack(fill=tk.BOTH, expand=True)

        self._build_joint_analysis_ui()

    def _build_joint_analysis_ui(self):
        # scroll region
        canvas = tk.Canvas(self.analysis_frame)
        scrollbar = ttk.Scrollbar(self.analysis_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._populate_joint_rows()

    def _clear_joint_ui_lists(self):
        self.value_labels = []
        self.velocity_labels = []
        self.limit_labels = []
        self.velocity_limit_labels = []
        self.status_labels = []
        self.velocity_status_labels = []

    def _populate_joint_rows(self):
        # clear existing widgets
        for w in list(self.scrollable_frame.winfo_children()):
            w.destroy()
        self._clear_joint_ui_lists()

        title_frame = ttk.Frame(self.scrollable_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(title_frame, text="关节名称", width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="位置值", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="位置状态", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="速度值", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="速度状态", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="位置限位", width=25).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="速度限位", width=15).pack(side=tk.LEFT, padx=5)

        for i, name in enumerate(self.joint_names):
            joint_main_frame = ttk.Frame(self.scrollable_frame)
            joint_main_frame.pack(fill=tk.X, pady=2)

            ttk.Label(joint_main_frame, text=name, width=20).pack(side=tk.LEFT, padx=5)

            value_label = ttk.Label(joint_main_frame, text="", width=10)
            value_label.pack(side=tk.LEFT, padx=5)
            self.value_labels.append(value_label)

            status_label = ttk.Label(joint_main_frame, text="", width=10)
            status_label.pack(side=tk.LEFT, padx=5)
            self.status_labels.append(status_label)

            velocity_label = ttk.Label(joint_main_frame, text="", width=10)
            velocity_label.pack(side=tk.LEFT, padx=5)
            self.velocity_labels.append(velocity_label)

            velocity_status_label = ttk.Label(joint_main_frame, text="", width=10)
            velocity_status_label.pack(side=tk.LEFT, padx=5)
            self.velocity_status_labels.append(velocity_status_label)

            limit_label = ttk.Label(joint_main_frame, text="", width=25, foreground="blue")
            limit_label.pack(side=tk.LEFT, padx=5)
            self.limit_labels.append(limit_label)

            velocity_limit_label = ttk.Label(joint_main_frame, text="", width=15, foreground="purple")
            velocity_limit_label.pack(side=tk.LEFT, padx=5)
            self.velocity_limit_labels.append(velocity_limit_label)

        contact_frame = ttk.Frame(self.scrollable_frame)
        contact_frame.pack(fill=tk.X, pady=10)

        ttk.Label(contact_frame, text="接触状态:", width=20).pack(side=tk.LEFT, padx=5)
        for _ in range(5):
            ttk.Label(contact_frame, text="", width=10).pack(side=tk.LEFT, padx=5)

        ttk.Label(contact_frame, text="左脚:").pack(side=tk.LEFT, padx=(0, 2))
        self.left_contact_entry = ttk.Entry(contact_frame, textvariable=self.left_contact_var, width=8)
        self.left_contact_entry.pack(side=tk.LEFT, padx=2)
        self.left_contact_entry.bind("<Return>", self.save_contact_state)

        ttk.Label(contact_frame, text="右脚:").pack(side=tk.LEFT, padx=(10, 2))
        self.right_contact_entry = ttk.Entry(contact_frame, textvariable=self.right_contact_var, width=8)
        self.right_contact_entry.pack(side=tk.LEFT, padx=2)
        self.right_contact_entry.bind("<Return>", self.save_contact_state)

        ttk.Button(contact_frame, text="保存修改", command=self.save_contact_state).pack(side=tk.LEFT, padx=10)
        ttk.Button(contact_frame, text="保存到文件", command=self.save_to_file).pack(side=tk.LEFT, padx=5)

    def _rebuild_joint_analysis_ui(self):
        self.mode_label.config(text=f"Joint Mode: {self.joint_mode}")
        self._populate_joint_rows()

    # --------------------- Folder / NPZ selection UI ---------------------
    def browse_root_dir(self):
        d = filedialog.askdirectory(initialdir=self.npz_root_dir or os.getcwd(), title="选择NPZ根目录")
        if not d:
            return
        self.root_dir_var.set(d)
        self.refresh_npz_tree()

    def refresh_npz_tree(self):
        root_dir = self.root_dir_var.get().strip()
        if not root_dir:
            messagebox.showwarning("提示", "请先填写/选择NPZ根目录")
            return
        if not os.path.isdir(root_dir):
            messagebox.showerror("路径错误", f"目录不存在:\n{root_dir}")
            return

        self.npz_root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.folder2files = _list_npz_files(self.npz_root_dir)

        # Treeview：目录可折叠 + 按需加载
        self._tree_build_root()

        # 若当前文件在树中，默认勾选它
        if self.current_npz_path and os.path.isfile(self.current_npz_path):
            self.checked_paths.add(self.current_npz_path)
            self._update_play_queue_from_checked()

        # 若没有 current，则自动加载第一个勾选/第一个找到的 npz
        if not self.current_npz_path:
            first = self.play_queue[0] if self.play_queue else self._get_first_npz_in_root()
            if first:
                self.load_npz(first)

    def _tree_build_root(self):
        if self.npz_tree is None:
            return

        self.npz_tree.delete(*self.npz_tree.get_children(""))
        self.npz_item2path.clear()

        # folder -> children folders/files 索引（用于按需加载）
        self._tree_children_folders: dict[str, set[str]] = {}
        self._tree_children_files: dict[str, list[str]] = {}

        for rel_folder, files in self.folder2files.items():
            rel_folder = "." if rel_folder in ("", ".") else rel_folder
            self._tree_children_files.setdefault(rel_folder, [])
            self._tree_children_files[rel_folder].extend(files)

            parts = [] if rel_folder == "." else rel_folder.split("/")
            cur = "."
            for p in parts:
                nxt = p if cur == "." else f"{cur}/{p}"
                self._tree_children_folders.setdefault(cur, set()).add(nxt)
                self._tree_children_folders.setdefault(nxt, set())
                cur = nxt

        root_id = self.npz_tree.insert("", "end", text=self._fmt_folder("."), open=True, tags=("folder",))
        self._tree_path2item = {".": root_id}

        # 预插入一级目录（不插入文件）
        for child in sorted(self._tree_children_folders.get(".", set())):
            self._tree_insert_folder(root_id, child)

        # 如果根目录本身就有 npz，插入 dummy 让根可展开后加载文件
        self.npz_tree.insert(root_id, "end", text="...", tags=("dummy",))

    def _fmt_folder(self, rel_folder: str) -> str:
        name = rel_folder if rel_folder != "." else "(root)"
        return f"📁 {name}"

    def _fmt_file(self, rel_path: str, checked: bool) -> str:
        box = "☑" if checked else "☐"
        return f"{box} {rel_path}"

    def _tree_insert_folder(self, parent_item: str, rel_folder: str):
        if rel_folder in self._tree_path2item:
            return self._tree_path2item[rel_folder]
        folder_item = self.npz_tree.insert(parent_item, "end", text=self._fmt_folder(rel_folder), open=False, tags=("folder",))
        self._tree_path2item[rel_folder] = folder_item
        self.npz_tree.insert(folder_item, "end", text="...", tags=("dummy",))
        return folder_item

    def on_tree_open(self, event=None):
        if self.npz_tree is None:
            return
        item = self.npz_tree.focus()
        if not item:
            return
        if "folder" not in set(self.npz_tree.item(item, "tags") or []):
            return

        # 若已加载（没有 dummy）则不重复加载
        children = self.npz_tree.get_children(item)
        if children and all("dummy" not in set(self.npz_tree.item(c, "tags") or []) for c in children):
            return

        # 找到该 folder 的 rel path
        rel_folder = "."
        for k, v in self._tree_path2item.items():
            if v == item:
                rel_folder = k
                break

        # 清理 dummy
        for c in list(children):
            if "dummy" in set(self.npz_tree.item(c, "tags") or []):
                self.npz_tree.delete(c)

        # 子目录
        for child_folder in sorted(self._tree_children_folders.get(rel_folder, set())):
            self._tree_insert_folder(item, child_folder)

        # 文件（仅该目录下）
        files = sorted(self._tree_children_files.get(rel_folder, []))
        for abs_path in files:
            rel_path = os.path.relpath(abs_path, self.npz_root_dir).replace("\\", "/")
            checked = abs_path in self.checked_paths
            file_item = self.npz_tree.insert(item, "end", text=self._fmt_file(rel_path, checked), tags=("file",))
            self.npz_item2path[file_item] = abs_path

    def on_tree_click(self, event):
        if self.npz_tree is None:
            return
        item = self.npz_tree.identify_row(event.y)
        if not item:
            return
        if "file" not in set(self.npz_tree.item(item, "tags") or []):
            return

        abs_path = self.npz_item2path.get(item)
        if not abs_path:
            return

        rel_path = os.path.relpath(abs_path, self.npz_root_dir).replace("\\", "/")
        if abs_path in self.checked_paths:
            self.checked_paths.remove(abs_path)
            self.npz_tree.item(item, text=self._fmt_file(rel_path, False))
        else:
            self.checked_paths.add(abs_path)
            self.npz_tree.item(item, text=self._fmt_file(rel_path, True))

        self._update_play_queue_from_checked()

        if not self.is_playing and self.play_queue:
            if self.current_npz_path != self.play_queue[0]:
                self.load_npz(self.play_queue[0])

    def _update_play_queue_from_checked(self):
        self.play_queue = sorted(self.checked_paths)
        if self.current_npz_path in self.play_queue:
            self.play_queue_idx = self.play_queue.index(self.current_npz_path)
        else:
            self.play_queue_idx = 0

    def tree_clear_all(self):
        self.checked_paths.clear()
        self._update_play_queue_from_checked()
        self._tree_refresh_visible_checks()

    def tree_select_all_expanded(self):
        # 仅全选“已展开且已加载到树里的文件”，避免一次性加载海量文件
        if self.npz_tree is None:
            return
        for item in self._iter_tree_items(""):
            if "file" in set(self.npz_tree.item(item, "tags") or []):
                abs_path = self.npz_item2path.get(item)
                if abs_path:
                    self.checked_paths.add(abs_path)
        self._update_play_queue_from_checked()
        self._tree_refresh_visible_checks()

    def tree_collapse_all(self):
        if self.npz_tree is None:
            return
        for item in self._iter_tree_items(""):
            if "folder" in set(self.npz_tree.item(item, "tags") or []):
                self.npz_tree.item(item, open=False)

    def tree_expand_all(self):
        # 文件非常多时仍可能比较重，所以按钮标注“谨慎”
        if self.npz_tree is None:
            return
        for item in self._iter_tree_items(""):
            if "folder" in set(self.npz_tree.item(item, "tags") or []):
                self.npz_tree.item(item, open=True)
                self.on_tree_open()

    def _iter_tree_items(self, parent):
        if self.npz_tree is None:
            return
        for c in self.npz_tree.get_children(parent):
            yield c
            yield from self._iter_tree_items(c)

    def _tree_refresh_visible_checks(self):
        if self.npz_tree is None:
            return
        for item in self._iter_tree_items(""):
            if "file" in set(self.npz_tree.item(item, "tags") or []):
                abs_path = self.npz_item2path.get(item)
                if not abs_path:
                    continue
                rel_path = os.path.relpath(abs_path, self.npz_root_dir).replace("\\", "/")
                self.npz_tree.item(item, text=self._fmt_file(rel_path, abs_path in self.checked_paths))

    # --------------------- Tag functions ---------------------

    def add_tag(self, event=None):
        tag_text = self.current_tag_text.get().strip()
        if not tag_text:
            messagebox.showwarning("输入错误", "请输入tag内容")
            return

        frame_num = self.current_frame
        if frame_num not in self.tags:
            self.tags[frame_num] = []

        if tag_text not in self.tags[frame_num]:
            self.tags[frame_num].append(tag_text)
            self.update_current_tags_display()
            self.current_tag_text.set("")
            messagebox.showinfo("添加成功", f"已为帧 {frame_num + 1} 添加Tag: {tag_text}")
        else:
            messagebox.showwarning("重复Tag", f"帧 {frame_num + 1} 已存在相同的Tag")

    def update_current_tags_display(self):
        frame_num = self.current_frame
        if frame_num in self.tags and self.tags[frame_num]:
            tags_text = ", ".join(self.tags[frame_num])
            self.current_tags_var.set(f"当前帧Tag: {tags_text}")
        else:
            self.current_tags_var.set("当前帧Tag: 无")

    def view_all_tags(self):
        if not self.tags:
            messagebox.showinfo("Tag信息", "还没有添加任何Tag")
            return

        tags_window = tk.Toplevel(self.root)
        tags_window.title("所有Tag信息")
        tags_window.geometry("600x400")

        main_frame = ttk.Frame(tags_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title_label = ttk.Label(main_frame, text=f"共 {len(self.tags)} 帧有Tag标记", font=("TkDefaultFont", 12, "bold"))
        title_label.pack(pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        sorted_frames = sorted(self.tags.keys())
        for frame_num in sorted_frames:
            frame_tag_frame = ttk.LabelFrame(scrollable_frame, text=f"帧 {frame_num + 1}")
            frame_tag_frame.pack(fill=tk.X, padx=5, pady=5)

            for tag in self.tags[frame_num]:
                ttk.Label(frame_tag_frame, text=f"• {tag}").pack(anchor=tk.W, padx=10, pady=2)

            ttk.Button(
                frame_tag_frame,
                text="删除该帧所有Tag",
                command=lambda fn=frame_num: self.delete_frame_tags(fn, tags_window),
            ).pack(pady=5)

        ttk.Button(main_frame, text="导出Tag报告", command=self.export_tags_report).pack(pady=10)

    def delete_frame_tags(self, frame_num, tags_window=None):
        if frame_num in self.tags:
            del self.tags[frame_num]
            if frame_num == self.current_frame:
                self.update_current_tags_display()
            if tags_window:
                tags_window.destroy()
                self.view_all_tags()
            messagebox.showinfo("删除成功", f"已删除帧 {frame_num + 1} 的所有Tag")

    def export_tags_report(self):
        try:
            save_dir = "storage/data/tag_reports"
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(self.file_name)[0] if self.file_name else "npz"
            save_path = f"{save_dir}/{base_name}_tags_report.txt"

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"NPZ文件: {self.file_name}\n")
                f.write(f"总帧数: {self.num_frames}\n")
                f.write(f"有Tag的帧数: {len(self.tags)}\n")
                f.write("=" * 50 + "\n\n")

                sorted_frames = sorted(self.tags.keys())
                for frame_num in sorted_frames:
                    f.write(f"帧 {frame_num + 1}:\n")
                    for tag in self.tags[frame_num]:
                        f.write(f"  - {tag}\n")
                    f.write("\n")

            messagebox.showinfo("导出成功", f"Tag报告已保存至:\n{save_path}")
        except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError) as e:
            messagebox.showerror("导出失败", f"保存报告时出错:\n{str(e)}")

    # --------------------- Core frame ops ---------------------
    def load_frame(self, frame_idx: int):
        if self.qpos_23d is None:
            return
        self.current_frame = int(frame_idx)
        self.frame_var.set(f"{self.current_frame + 1}/{self.num_frames}")
        if self.frame_slider is not None:
            self.frame_slider.set(self.current_frame)

        self.mode_label.config(text=f"Joint Mode: {self.joint_mode}")

        self.update_model()

        for j in range(len(self.joint_names)):
            pos_value = self.qpos_23d[self.current_frame, 7 + j]
            vel_value = self.velocities[self.current_frame, 7 + j]
            self.update_value_display(j, pos_value, vel_value)

        self.load_contact_values()
        self.update_current_tags_display()

        if self.viewer:
            self.viewer.sync()

    def update_value_display(self, joint_idx, pos_value, vel_value):
        self.value_labels[joint_idx].config(text=f"{pos_value:.4f}")
        self.velocity_labels[joint_idx].config(text=f"{vel_value:.4f}")

        pos_lower = self.lower_bounds[joint_idx]
        pos_upper = self.upper_bounds[joint_idx]

        if pos_value < pos_lower:
            self.status_labels[joint_idx].config(text="↓超下限", foreground="red")
        elif pos_value > pos_upper:
            self.status_labels[joint_idx].config(text="↑超上限", foreground="red")
        else:
            self.status_labels[joint_idx].config(text="正常", foreground="green")

        vel_limit = self.velocity_limits[joint_idx]
        if abs(vel_value) > vel_limit:
            self.velocity_status_labels[joint_idx].config(text="超限", foreground="red")
        else:
            self.velocity_status_labels[joint_idx].config(text="正常", foreground="green")

        self.limit_labels[joint_idx].config(text=f"[{pos_lower:.4f}, {pos_upper:.4f}]", foreground="blue")
        self.velocity_limit_labels[joint_idx].config(text=f"±{vel_limit:.1f}", foreground="purple")

    def load_contact_values(self):
        if self.modified_foot_contact is not None and self.current_frame < self.num_frames:
            left_contact = self.modified_foot_contact[self.current_frame, 0]
            right_contact = self.modified_foot_contact[self.current_frame, 1]
            self.left_contact_var.set(f"{left_contact:.2f}")
            self.right_contact_var.set(f"{right_contact:.2f}")

    def save_contact_state(self, event=None):
        if self.modified_foot_contact is None:
            return
        try:
            left_contact = float(self.left_contact_var.get())
            right_contact = float(self.right_contact_var.get())
            self.modified_foot_contact[self.current_frame, 0] = left_contact
            self.modified_foot_contact[self.current_frame, 1] = right_contact
            self.contact_modified = True
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字")
            self.load_contact_values()

    def save_to_file(self):
        if not self.contact_modified:
            messagebox.showinfo("提示", "接触状态未被修改，无需保存")
            return
        if self.npzdata is None:
            return

        try:
            save_data = {}
            for key in self.npzdata.keys():
                if key == "foot_contact":
                    save_data[key] = self.modified_foot_contact
                else:
                    save_data[key] = self.npzdata[key]

            save_dir = "storage/data/mocap/fc_modify"
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(self.file_name)[0]
            save_filename = f"{save_dir}/{base_name}_modified.npz"

            np.savez(save_filename, **save_data)
            messagebox.showinfo("保存成功", f"数据已保存到: {save_filename}")
            self.contact_modified = False
        except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError) as e:
            messagebox.showerror("保存错误", f"保存文件时出错: {str(e)}")

    # --------------------- Navigation ---------------------
    def prev_frame(self):
        if self.qpos_23d is None:
            return
        if self.current_frame > 0:
            self.current_frame -= 1
            self.load_frame(self.current_frame)

    def next_frame(self):
        if self.qpos_23d is None:
            return
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.load_frame(self.current_frame)

    def go_to_frame(self):
        if self.qpos_23d is None:
            return
        try:
            frame_idx = int(self.frame_entry_var.get()) - 1
            if 0 <= frame_idx < self.num_frames:
                self.load_frame(frame_idx)
            else:
                messagebox.showerror("无效帧号", f"帧号必须在1到{self.num_frames}之间")
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的帧号")
        finally:
            self.frame_entry_var.set("")

    def on_frame_slider(self, value):
        if self.qpos_23d is None:
            return
        frame_idx = int(float(value))
        if frame_idx != self.current_frame:
            self.load_frame(frame_idx)

    # --------------------- Play (auto-advance between checked NPZs) ---------------------
    def toggle_play(self):
        if not self.is_playing:
            self._update_play_queue_from_checked()
            if not self.play_queue and self.current_npz_path:
                # No checked files -> treat current as a single-file play
                self.play_queue = [self.current_npz_path]
                self.play_queue_idx = 0

            if not self.play_queue:
                messagebox.showwarning("提示", "请先勾选至少一个NPZ再播放")
                return

            # Ensure current is one of them
            if self.current_npz_path not in self.play_queue:
                self.play_queue_idx = 0
                self.load_npz(self.play_queue[0])

            self.is_playing = True
            self.play_button.config(text="停止")
            self.play_animation()
        else:
            self.is_playing = False
            self.play_button.config(text="播放")

    def _advance_to_next_npz(self) -> bool:
        self._update_play_queue_from_checked()
        if not self.play_queue:
            return False

        # Move to next file (cyclic)
        if self.current_npz_path in self.play_queue:
            idx = self.play_queue.index(self.current_npz_path)
        else:
            idx = -1
        next_idx = (idx + 1) % len(self.play_queue)
        next_path = self.play_queue[next_idx]

        if next_path != self.current_npz_path:
            self.load_npz(next_path)
        else:
            self.load_frame(0)
        return True

    def play_animation(self):
        if not self.is_playing:
            return
        if self.qpos_23d is None:
            self.is_playing = False
            self.play_button.config(text="播放")
            return

        if self.current_frame >= self.num_frames:
            # finished current file -> next checked file
            ok = self._advance_to_next_npz()
            if not ok:
                self.is_playing = False
                self.play_button.config(text="播放")
                return

        self.load_frame(self.current_frame)
        self.current_frame += 1

        if self.is_playing:
            self.root.after(20, self.play_animation)  # 50Hz

    # --------------------- Model update ---------------------
    def update_model(self):
        if self.qpos_23d is None:
            return

        if self.joint_mode == ['66155']:
            self.qpos_29d[self.current_frame, :7] = self.qpos_23d[self.current_frame, :7]
            valid_indices_29d = [i for i in range(0, 29) if i not in [13, 14, 20, 21, 27, 28]]
            self.qpos_29d[self.current_frame, valid_indices_29d] = self.qpos_23d[self.current_frame, 7:]
            self.data.qpos[:] = np.concatenate((self.root_msg[self.current_frame], self.qpos_29d[self.current_frame]))
        elif self.joint_mode == ['66377']:
            self.data.qpos[:] = self.qpos_23d[self.current_frame]

        mujoco.mj_forward(self.model, self.data)

        if self.camera_follow:
            pelvis_pos = self.data.body("pelvis").xpos
            torso_pos = self.data.body("torso_link").xpos
            forward_direction = torso_pos - pelvis_pos
            forward_direction[2] = 0
            n = np.linalg.norm(forward_direction)
            if n > 1e-6:
                forward_direction /= n
            else:
                forward_direction[:] = np.array([1.0, 0.0, 0.0])

            camera_distance = 2.5
            camera_pos = pelvis_pos - forward_direction * camera_distance
            camera_pos[2] = pelvis_pos[2] + 0.5

            lookat_pos = pelvis_pos.copy()
            lookat_pos[2] += 0.2

            self.viewer.cam.lookat[:] = lookat_pos
            self.viewer.cam.distance = camera_distance
            self.viewer.cam.elevation = -10
            self.viewer.cam.azimuth = 0

    def on_camera_follow_change(self):
        self.camera_follow = self.camera_follow_var.get()
        if hasattr(self, "current_frame"):
            self.update_model()

    # --------------------- Plot ---------------------
    def plot_joint_trajectories(self):
        if self.qpos_23d is None:
            return
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 5, figsize=(15, 12))
        axes = axes.flatten()

        for i, joint_name in enumerate(self.joint_names[:25]):
            if i >= len(axes):
                break
            joint_positions = self.qpos_23d[:, 7 + i]
            ax = axes[i]
            ax.plot(joint_positions)
            ax.set_title(joint_name)
            ax.axvline(x=self.current_frame, color="r", linestyle="--", alpha=0.7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # --------------------- Run ---------------------
    def run(self):
        self.root.mainloop()
        if self.viewer:
            self.viewer.close()


if __name__ == "__main__":
    # 用法：
    # 1) 直接指定单个npz:
    #    analyzer = NPZAnalyzer(npz_path="data/mocap/a.npz")
    # 2) 指定根目录(递归扫描npz)，在左侧勾选并播放:
    #    analyzer = NPZAnalyzer(npz_root_dir="data/mocap")
    analyzer = NPZAnalyzer(
        npz_root_dir="data/mocap"
    )
    analyzer.run()
