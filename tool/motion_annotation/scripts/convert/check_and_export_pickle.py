import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import mujoco
import mujoco.viewer
import os
import time
import pickle
import sys
from types import ModuleType


# Patch sys.modules to fake missing modules from numpy 2.x
class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)


# Patch potentially missing modules
sys.modules['numpy._core'] = FakeModule('numpy._core', np.core if hasattr(np, 'core') else np)
sys.modules['numpy._core.multiarray'] = FakeModule('numpy._core.multiarray', getattr(np.core, 'multiarray', None))


class NPZAnalyzer:
    def __init__(self, npz_path, xml_path):
        # 初始化pkl文件相关变量
        self.pkl_files = []
        self.current_pkl_index = 0
        self.pkl_data = None
        self.folder_path = None
        self.frequency = 50  # 默认50Hz

        # 如果没有提供npz_path，进入pkl模式
        if not npz_path:
            self.is_pkl_mode = True
            self.qpos_23d = None
            self.root_msg = None
            self.num_frames = 0
        else:
            self.is_pkl_mode = False
            self.npzdata = np.load(npz_path, allow_pickle=True)
            self.file_name = os.path.basename(npz_path)
            self.qpos_23d = self.npzdata["qpos"].copy()
            self.root_msg = self.npzdata["qpos"][:, :7].copy()
            self.num_frames, self.num_joints = self.qpos_23d.shape

        self.camera_follow = False
        self.play_speed = 1

        # 删除star_chain_navi相关代码
        self.star_chain_navi = None

        # 删除foot_contact相关代码
        self.foot_contact = None

        # 删除interp相关代码
        self.contact_modified = False

        # 初始化关节模式
        self.joint_mode = None
        if not self.is_pkl_mode:
            if (self.num_joints - 7) == 23:
                self.joint_mode = ['66155']
            elif (self.num_joints - 7) == 29:
                self.joint_mode = ['66377']
            else:
                print("Invalid joint numbers!")
                exit(0)
        else:
            # 默认为66155模式
            self.joint_mode = ['66377']

        # 计算速度 (rad/s)
        self.dt = 1 / 50
        self.velocities = None
        if not self.is_pkl_mode:
            self.velocities = np.zeros_like(self.qpos_23d)
            self.velocities[1:] = (self.qpos_23d[1:] - self.qpos_23d[:-1]) / self.dt
            self.velocities[0] = 0

        self.plot_windows = []

        # 创建完整的29维qpos
        self.qpos_29d = None
        if not self.is_pkl_mode:
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
        else:
            self.qpos_29d = None

        # 创建MuJoCo环境
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # 初始化界面
        self.root = tk.Tk()
        self.root.title("NPZ/PKL 超限检测工具")
        self.root.geometry("1800x700")
        self.current_frame = 0

        self.joint_names = []
        self.lower_bounds = []
        self.upper_bounds = []

        # 遍历模型中的所有关节，读取它们的名称和范围
        for i in range(self.model.njnt):
            jnt_name = self.model.joint(i).name
            jnt_range = self.model.jnt_range[i]

            # 只处理有名称的关节（排除自由关节等）
            if jnt_name and jnt_name != "floating_base_joint":
                self.joint_names.append(jnt_name)
                self.lower_bounds.append(jnt_range[0])  # 下限
                self.upper_bounds.append(jnt_range[1])  # 上限

        print(f"从XML模型加载了 {len(self.joint_names)} 个关节的范围信息")

        # 根据关节模式过滤关节
        if self.joint_mode == ['66155']:
            exclude_patterns = [
                'waist_roll', 'waist_pitch',
                'left_wrist_pitch', 'left_wrist_yaw',
                'right_wrist_pitch', 'right_wrist_yaw'
            ]

            # 过滤关节，只保留不在排除列表中的关节
            filtered_joint_names = []
            filtered_lower_bounds = []
            filtered_upper_bounds = []

            for name, lower, upper in zip(self.joint_names, self.lower_bounds, self.upper_bounds):
                exclude = False
                for pattern in exclude_patterns:
                    if pattern in name:
                        exclude = True
                        break
                if not exclude:
                    filtered_joint_names.append(name)
                    filtered_lower_bounds.append(lower)
                    filtered_upper_bounds.append(upper)

            self.joint_names = filtered_joint_names
            self.lower_bounds = filtered_lower_bounds
            self.upper_bounds = filtered_upper_bounds

            print(f"66155模式过滤后保留 {len(self.joint_names)} 个关节")

        elif self.joint_mode == ['66377']:
            print(f"66377模式使用所有 {len(self.joint_names)} 个关节")

        # 检查关节数量是否匹配
        expected_joints = 23 if self.joint_mode == ['66155'] else 29
        if len(self.joint_names) != expected_joints:
            print(f"警告: XML中读取的关节数({len(self.joint_names)})与预期({expected_joints})不匹配")
            # 可以选择截断或填充，这里选择截断
            if len(self.joint_names) > expected_joints:
                self.joint_names = self.joint_names[:expected_joints]
                self.lower_bounds = self.lower_bounds[:expected_joints]
                self.upper_bounds = self.upper_bounds[:expected_joints]
                print(f"已截断为前{expected_joints}个关节")

        self.velocity_limits = [
            32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
            32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
            32.0, 37.0, 37.0,
            37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0,
            37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0
        ]

        self.overlimit_records = []
        self.is_playing = False
        self.play_start_frame = 0
        self.play_completed = False

        # 初始化UI
        self.setup_ui()

        # 如果是pkl模式，加载文件夹选择界面
        if self.is_pkl_mode:
            self.select_folder()

        # 加载第一帧
        if self.num_frames > 0:
            self.load_frame(0)

    def quaternion_xyzw_to_wxyz(self, quat_xyzw):
        """
        将四元数从XYZW格式转换为WXYZ格式
        Args:
            quat_xyzw: 形状为(4,)或(n,4)的数组，XYZW格式的四元数
        Returns:
            quat_wxyz: WXYZ格式的四元数
        """
        if len(quat_xyzw.shape) == 1:
            # 单个四元数 [x, y, z, w] -> [w, x, y, z]
            return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        else:
            # 多个四元数 (n, 4)
            quat_wxyz = np.zeros_like(quat_xyzw)
            quat_wxyz[:, 0] = quat_xyzw[:, 3]  # w
            quat_wxyz[:, 1] = quat_xyzw[:, 0]  # x
            quat_wxyz[:, 2] = quat_xyzw[:, 1]  # y
            quat_wxyz[:, 3] = quat_xyzw[:, 2]  # z
            return quat_wxyz

    def setup_ui(self):
        self.frame_var = tk.StringVar(value=f"{self.current_frame + 1}/{self.num_frames}")
        self.camera_follow_var = tk.BooleanVar(value=self.camera_follow)
        self.speed_var = tk.StringVar(value="1x")

        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 顶部控制面板
        control_frame = ttk.LabelFrame(main_frame, text="帧控制")
        control_frame.pack(fill=tk.X, pady=10)

        mode_label = ttk.Label(control_frame, text=f"Joint Mode: {self.joint_mode}", font=("TkDefaultFont", 10, "bold"))
        mode_label.pack(side=tk.LEFT, padx=10)

        self.camera_follow_cb = ttk.Checkbutton(
            control_frame, text="相机跟随", variable=self.camera_follow_var, command=self.on_camera_follow_change
        )
        self.camera_follow_cb.pack(side=tk.LEFT, padx=10)

        speed_label = ttk.Label(control_frame, text="倍速:")
        speed_label.pack(side=tk.LEFT, padx=5)

        speed_box = ttk.Combobox(
            control_frame,
            textvariable=self.speed_var,
            values=["1x", "2x", "5x"],
            width=4,
            state="readonly"
        )
        speed_box.pack(side=tk.LEFT, padx=5)
        speed_box.bind("<<ComboboxSelected>>", self.on_speed_change)

        # 帧导航
        frame_label = ttk.Label(control_frame, text="当前帧:")
        frame_label.pack(side=tk.LEFT, padx=5)

        frame_display = ttk.Label(control_frame, textvariable=self.frame_var)
        frame_display.pack(side=tk.LEFT, padx=5)

        prev_btn = ttk.Button(control_frame, text="上一帧", command=self.prev_frame)
        prev_btn.pack(side=tk.LEFT, padx=5)

        next_btn = ttk.Button(control_frame, text="下一帧", command=self.next_frame)
        next_btn.pack(side=tk.LEFT, padx=5)

        self.play_button = ttk.Button(control_frame, text="播放", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.frame_slider = ttk.Scale(
            control_frame, from_=0, to=self.num_frames - 1, orient=tk.HORIZONTAL, length=300,
            command=self.on_frame_slider
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # PKL文件控制区域
        if self.is_pkl_mode:
            pkl_frame = ttk.Frame(control_frame)
            pkl_frame.pack(side=tk.LEFT, padx=10)

            select_folder_btn = ttk.Button(pkl_frame, text="选择文件夹", command=self.select_folder)
            select_folder_btn.pack(side=tk.LEFT, padx=5)

            prev_pkl_btn = ttk.Button(pkl_frame, text="上一个PKL", command=self.prev_pkl)
            prev_pkl_btn.pack(side=tk.LEFT, padx=5)

            next_pkl_btn = ttk.Button(pkl_frame, text="下一个PKL", command=self.next_pkl)
            next_pkl_btn.pack(side=tk.LEFT, padx=5)

            export_btn = ttk.Button(pkl_frame, text="导出NPZ", command=self.export_to_npz)
            export_btn.pack(side=tk.LEFT, padx=5)

        # 关节分析区域
        analysis_frame = ttk.LabelFrame(main_frame, text="关节分析")
        analysis_frame.pack(fill=tk.BOTH, expand=True)

        # 创建滚动区域
        canvas = tk.Canvas(analysis_frame)
        scrollbar = ttk.Scrollbar(analysis_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 为每个关节创建显示面板
        self.value_labels = []
        self.velocity_labels = []
        self.limit_labels = []
        self.velocity_limit_labels = []
        self.status_labels = []
        self.velocity_status_labels = []

        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(title_frame, text="关节名称", width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="位置值", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="位置状态", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="速度值", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="速度状态", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="位置限位", width=25).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="速度限位", width=15).pack(side=tk.LEFT, padx=5)

        # 创建关节行
        for i, name in enumerate(self.joint_names):
            joint_main_frame = ttk.Frame(scrollable_frame)
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

    def select_folder(self):
        """选择包含pkl文件的文件夹"""
        self.folder_path = filedialog.askdirectory(title="选择包含PKL文件的文件夹")
        if self.folder_path:
            # 获取所有pkl文件
            self.pkl_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pkl')]
            self.pkl_files.sort()

            if self.pkl_files:
                self.current_pkl_index = 0
                self.load_pkl_file()
            else:
                messagebox.showinfo("提示", "选择的文件夹中没有找到pkl文件")

    def load_pkl_file(self):
        """加载当前索引的pkl文件"""
        if not self.pkl_files:
            return

        file_path = os.path.join(self.folder_path, self.pkl_files[self.current_pkl_index])

        try:
            with open(file_path, 'rb') as f:
                self.pkl_data = pickle.load(f)

            print(f"Successfully loaded {self.pkl_files[self.current_pkl_index]}")

            # 读取root_pos、root_rot和dof_pos
            root_pos = self.pkl_data['root_pos']
            root_rot_xyzw = self.pkl_data['root_rot']  # 这是XYZW格式
            dof_pos = self.pkl_data['dof_pos']

            # 确保维度正确
            assert root_pos.shape[0] == root_rot_xyzw.shape[0] == dof_pos.shape[0], "数据长度不匹配"

            # 将四元数从XYZW格式转换为WXYZ格式
            root_rot_wxyz = self.quaternion_xyzw_to_wxyz(root_rot_xyzw)

            print(f"四元数已从XYZW转换为WXYZ格式，前5帧示例:")
            for i in range(min(5, len(root_rot_xyzw))):
                print(f"帧{i}: XYZW {root_rot_xyzw[i]} -> WXYZ {root_rot_wxyz[i]}")

            # 拼接root_pos (3维), root_rot_wxyz (4维, WXYZ格式), dof_pos (23或29维)
            self.qpos_23d = np.hstack((root_pos, root_rot_wxyz, dof_pos))
            self.root_msg = self.qpos_23d[:, :7].copy()

            # 更新帧信息
            self.num_frames, self.num_joints = self.qpos_23d.shape

            # 读取fps
            if 'fps' in self.pkl_data:
                self.frequency = self.pkl_data['fps']
                self.dt = 1 / self.frequency
            else:
                self.frequency = 50  # 默认50Hz
                self.dt = 1 / 50

            # 计算速度
            self.velocities = np.zeros_like(self.qpos_23d)
            self.velocities[1:] = (self.qpos_23d[1:] - self.qpos_23d[:-1]) / self.dt
            self.velocities[0] = 0

            # 初始化29维qpos
            self.qpos_29d = np.zeros((self.num_frames, 29))

            # 更新帧滑块
            self.frame_slider.config(to=self.num_frames - 1)

            # 加载第一帧
            self.load_frame(0)

        except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError) as e:
            messagebox.showerror("错误", f"加载pkl文件失败: {str(e)}")

    def prev_pkl(self):
        """加载上一个pkl文件"""
        if self.pkl_files and self.current_pkl_index > 0:
            self.current_pkl_index -= 1
            self.load_pkl_file()

    def next_pkl(self):
        """加载下一个pkl文件"""
        if self.pkl_files and self.current_pkl_index < len(self.pkl_files) - 1:
            self.current_pkl_index += 1
            self.load_pkl_file()

    def export_to_npz(self):
        """将当前pkl数据导出为npz文件"""
        if not self.is_pkl_mode or self.qpos_23d is None:
            messagebox.showinfo("提示", "没有可导出的数据")
            return

        try:
            # 准备导出数据
            export_data = {
                'qpos': self.qpos_23d,
                'frequence': self.frequency,
            }

            # 设置导出文件名
            base_name = os.path.splitext(self.pkl_files[self.current_pkl_index])[0]
            # 修改导出路径为工程根目录下的npz文件夹
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            npz_folder = os.path.join(project_root, "npz")

            # 创建npz文件夹（如果不存在）
            if not os.path.exists(npz_folder):
                os.makedirs(npz_folder)

            export_path = os.path.join(npz_folder, f"{base_name}.npz")

            # 导出
            np.savez(export_path, **export_data)
            messagebox.showinfo("成功", f"数据已导出到: {export_path}")

        except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError) as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")

    def on_speed_change(self, event=None):
        speed_text = self.speed_var.get()
        if speed_text == "1x":
            self.play_speed = 1
        elif speed_text == "2x":
            self.play_speed = 2
        elif speed_text == "5x":
            self.play_speed = 5

    def load_frame(self, frame_idx):
        """加载指定帧的关节数据"""
        if self.num_frames == 0:
            return

        self.current_frame = frame_idx
        self.frame_var.set(f"{self.current_frame + 1}/{self.num_frames}")
        self.frame_slider.set(self.current_frame)

        # 更新模型
        self.update_model()

        # 更新关节显示
        for j in range(len(self.joint_names)):
            pos_value = self.qpos_23d[frame_idx, 7 + j]
            vel_value = self.velocities[frame_idx, 7 + j]
            self.update_value_display(j, pos_value, vel_value)

        if self.viewer:
            self.viewer.sync()

    def update_value_display(self, joint_idx, pos_value, vel_value):
        """更新关节值和速度的显示"""
        # 更新位置值
        self.value_labels[joint_idx].config(text=f"{pos_value:.4f}")

        # 更新速度值
        self.velocity_labels[joint_idx].config(text=f"{vel_value:.4f}")

        # 检查位置限位
        pos_lower = self.lower_bounds[joint_idx]
        pos_upper = self.upper_bounds[joint_idx]

        if pos_value < pos_lower:
            self.status_labels[joint_idx].config(text="↓超下限", foreground="red")
        elif pos_value > pos_upper:
            self.status_labels[joint_idx].config(text="↑超上限", foreground="red")
        else:
            self.status_labels[joint_idx].config(text="正常", foreground="green")

        # 检查速度限位
        vel_limit = self.velocity_limits[joint_idx]
        if abs(vel_value) > vel_limit:
            self.velocity_status_labels[joint_idx].config(text="超限", foreground="red")
        else:
            self.velocity_status_labels[joint_idx].config(text="正常", foreground="green")

        # 更新限位值显示
        self.limit_labels[joint_idx].config(text=f"[{pos_lower:.4f}, {pos_upper:.4f}]", foreground="blue")
        self.velocity_limit_labels[joint_idx].config(text=f"±{vel_limit:.1f}", foreground="purple")

    # 导航功能
    def prev_frame(self):
        """上一帧"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.load_frame(self.current_frame)

    def next_frame(self):
        """下一帧"""
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.load_frame(self.current_frame)

    def on_frame_slider(self, value):
        """滑块拖动事件"""
        frame_idx = int(float(value))
        if frame_idx != self.current_frame:
            self.load_frame(frame_idx)

    # 播放功能
    def toggle_play(self):
        """切换播放状态"""
        if not self.is_playing:
            self.is_playing = True
            self.play_button.config(text="停止")
            self.play_animation()
        else:
            self.is_playing = False
            self.play_button.config(text="播放")

    def play_animation(self):
        """播放动画"""
        if not self.is_playing or self.num_frames == 0:
            return

        if self.current_frame >= self.num_frames - 1:
            self.current_frame = 0

        self.load_frame(self.current_frame)
        self.current_frame += self.play_speed

        if self.current_frame >= self.num_frames:
            self.current_frame = 0

        delay = int(20 / self.play_speed)  # 1×=20ms, 2×=10ms, 5×=4ms

        if self.is_playing:
            self.root.after(delay, self.play_animation)  # 50Hz
        else:
            self.is_playing = False
            self.play_button.config(text="播放")

    # 模型更新
    def update_model(self):
        """更新模型显示"""
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
            forward_direction /= np.linalg.norm(forward_direction)

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
        """相机跟随状态变更"""
        self.camera_follow = self.camera_follow_var.get()
        if hasattr(self, 'current_frame'):
            self.update_model()

    def run(self):
        """运行分析器"""
        self.root.mainloop()
        if self.viewer:
            self.viewer.close()


if __name__ == "__main__":
    # 启动时不提供npz_path，进入pkl模式
    analyzer = NPZAnalyzer("",
                           "assets/unitree_g1/scene_mjx_dance_debug.xml")
    analyzer.run()
