import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import mujoco
import mujoco.viewer
import os
import time


class NPZAnalyzer:
    def __init__(self, npz_path):
        self.npzdata = np.load(npz_path, allow_pickle=True)
        self.file_name = os.path.basename(npz_path)
        self.camera_follow = False

        self.tags = {}

        self.qpos_23d = self.npzdata["qpos"].copy()
        self.root_msg = self.npzdata["qpos"][:, :7].copy()
        self.foot_contact = None
        if "foot_contact" in self.npzdata:
            self.foot_contact = self.npzdata["foot_contact"]
            print(f"Loaded foot contact data with shape: {self.foot_contact.shape}")
        else:
            print("Warning: No foot contact data found in NPZ file")

        # 删除interp相关代码
        self.contact_modified = False
        self.modified_foot_contact = self.foot_contact.copy() if self.foot_contact is not None else None
        self.num_frames, self.num_joints = self.qpos_23d.shape
        self.joint_mode = None
        if (self.num_joints - 7) == 23:
            self.joint_mode = ['66155']
        elif (self.num_joints - 7) == 29:
            self.joint_mode = ['66377']
        else:
            print("Invalid joint numbers!")
            exit(0)

        # 计算速度 (rad/s)
        self.dt = 1 / 50
        self.velocities = np.zeros_like(self.qpos_23d)
        self.velocities[1:] = (self.qpos_23d[1:] - self.qpos_23d[:-1]) / self.dt
        self.velocities[0] = 0

        self.plot_windows = []

        # 创建完整的29维qpos
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
        # 创建MuJoCo环境
        self.model = mujoco.MjModel.from_xml_path("assets/unitree_g1/scene_mjx.xml")
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # 初始化界面
        self.root = tk.Tk()
        self.root.title("NPZ 超限检测工具")
        self.root.geometry("1400x700")
        self.current_frame = 0

        # 定义关节名称和范围
        if self.joint_mode == ['66155']:
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
        elif self.joint_mode == ['66377']:
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

        self.overlimit_records = []
        self.is_playing = False
        self.play_start_frame = 0
        self.play_completed = False

        # 初始化UI
        self.setup_ui()
        self.load_frame(0)

    def setup_ui(self):
        self.current_tag_text = tk.StringVar(value="")
        self.current_tags_var = tk.StringVar(value="当前帧Tag: 无")
        self.frame_var = tk.StringVar(value=f"{self.current_frame + 1}/{self.num_frames}")
        self.camera_follow_var = tk.BooleanVar(value=self.camera_follow)
        self.frame_entry_var = tk.StringVar()
        self.left_contact_var = tk.StringVar()
        self.right_contact_var = tk.StringVar()

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

        plot_btn = ttk.Button(control_frame, text="绘制关节轨迹", command=self.plot_joint_trajectories)
        plot_btn.pack(side=tk.LEFT, padx=5)

        self.frame_slider = ttk.Scale(
            control_frame, from_=0, to=self.num_frames - 1, orient=tk.HORIZONTAL, length=300, command=self.on_frame_slider
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # 帧号输入框
        frame_entry_frame = ttk.Frame(control_frame)
        frame_entry_frame.pack(side=tk.LEFT, padx=5)

        frame_entry = ttk.Entry(frame_entry_frame, textvariable=self.frame_entry_var, width=8)
        frame_entry.pack(side=tk.LEFT)

        frame_set_btn = ttk.Button(frame_entry_frame, text="跳转", width=4, command=self.go_to_frame)
        frame_set_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Tag功能区域
        tag_frame = ttk.Frame(control_frame)
        tag_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(tag_frame, text="Tag:").pack(side=tk.LEFT, padx=(0, 2))
        self.tag_entry = ttk.Entry(tag_frame, textvariable=self.current_tag_text, width=15)
        self.tag_entry.pack(side=tk.LEFT, padx=2)
        self.tag_entry.bind('<Return>', self.add_tag)

        add_tag_btn = ttk.Button(tag_frame, text="添加Tag", command=self.add_tag)
        add_tag_btn.pack(side=tk.LEFT, padx=5)

        self.tags_label = ttk.Label(tag_frame, textvariable=self.current_tags_var)
        self.tags_label.pack(side=tk.LEFT, padx=5)

        view_tags_btn = ttk.Button(tag_frame, text="查看所有Tag", command=self.view_all_tags)
        view_tags_btn.pack(side=tk.LEFT, padx=5)

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

        # 添加接触状态行（在关节行下方）
        contact_frame = ttk.Frame(scrollable_frame)
        contact_frame.pack(fill=tk.X, pady=10)

        ttk.Label(contact_frame, text="接触状态:", width=20).pack(side=tk.LEFT, padx=5)

        # 占位符对齐
        for _ in range(5):
            ttk.Label(contact_frame, text="", width=10).pack(side=tk.LEFT, padx=5)

        ttk.Label(contact_frame, text="左脚:").pack(side=tk.LEFT, padx=(0, 2))
        self.left_contact_entry = ttk.Entry(contact_frame, textvariable=self.left_contact_var, width=8)
        self.left_contact_entry.pack(side=tk.LEFT, padx=2)
        self.left_contact_entry.bind('<Return>', self.save_contact_state)

        ttk.Label(contact_frame, text="右脚:").pack(side=tk.LEFT, padx=(10, 2))
        self.right_contact_entry = ttk.Entry(contact_frame, textvariable=self.right_contact_var, width=8)
        self.right_contact_entry.pack(side=tk.LEFT, padx=2)
        self.right_contact_entry.bind('<Return>', self.save_contact_state)

        ttk.Button(contact_frame, text="保存修改", command=self.save_contact_state).pack(side=tk.LEFT, padx=10)
        ttk.Button(contact_frame, text="保存到文件", command=self.save_to_file).pack(side=tk.LEFT, padx=5)

    # Tag相关功能
    def add_tag(self, event=None):
        """为当前帧添加tag"""
        tag_text = self.current_tag_text.get().strip()
        if not tag_text:
            messagebox.showwarning("输入错误", "请输入tag内容")
            return

        frame_num = self.current_frame

        if frame_num not in self.tags:
            self.tags[frame_num] = []

        if tag_text not in self.tags[frame_num]:
            self.tags[frame_num].append(tag_text)
            print(f"帧号: {frame_num + 1}, Tag: {tag_text}")
            self.update_current_tags_display()
            self.current_tag_text.set("")
            messagebox.showinfo("添加成功", f"已为帧 {frame_num + 1} 添加Tag: {tag_text}")
        else:
            messagebox.showwarning("重复Tag", f"帧 {frame_num + 1} 已存在相同的Tag")

    def update_current_tags_display(self):
        """更新当前帧tag显示"""
        frame_num = self.current_frame
        if frame_num in self.tags and self.tags[frame_num]:
            tags_text = ", ".join(self.tags[frame_num])
            self.current_tags_var.set(f"当前帧Tag: {tags_text}")
        else:
            self.current_tags_var.set("当前帧Tag: 无")

    def view_all_tags(self):
        """查看所有tag的窗口"""
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

        # 创建滚动区域
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 按帧号排序显示tag
        sorted_frames = sorted(self.tags.keys())
        for frame_num in sorted_frames:
            frame_tag_frame = ttk.LabelFrame(scrollable_frame, text=f"帧 {frame_num + 1}")
            frame_tag_frame.pack(fill=tk.X, padx=5, pady=5)

            for tag in self.tags[frame_num]:
                ttk.Label(frame_tag_frame, text=f"• {tag}").pack(anchor=tk.W, padx=10, pady=2)

            ttk.Button(frame_tag_frame, text="删除该帧所有Tag", command=lambda fn=frame_num: self.delete_frame_tags(fn, tags_window)).pack(pady=5)

        ttk.Button(main_frame, text="导出Tag报告", command=self.export_tags_report).pack(pady=10)

    def delete_frame_tags(self, frame_num, tags_window=None):
        """删除指定帧的所有tag"""
        if frame_num in self.tags:
            del self.tags[frame_num]
            if frame_num == self.current_frame:
                self.update_current_tags_display()
            if tags_window:
                tags_window.destroy()
                self.view_all_tags()
            messagebox.showinfo("删除成功", f"已删除帧 {frame_num + 1} 的所有Tag")

    def export_tags_report(self):
        """导出tag报告到文件"""
        try:
            save_dir = "storage/data/tag_reports"
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(self.file_name)[0]
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

    # 基本功能
    def load_frame(self, frame_idx):
        """加载指定帧的关节数据"""
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

        # 加载接触状态
        self.load_contact_values()
        self.update_current_tags_display()

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

    def load_contact_values(self):
        """加载当前帧的接触状态"""
        if self.modified_foot_contact is not None and self.current_frame < self.num_frames:
            left_contact = self.modified_foot_contact[self.current_frame, 0]
            right_contact = self.modified_foot_contact[self.current_frame, 1]
            self.left_contact_var.set(f"{left_contact:.2f}")
            self.right_contact_var.set(f"{right_contact:.2f}")

    def save_contact_state(self, event=None):
        """保存接触状态修改"""
        try:
            left_contact = float(self.left_contact_var.get())
            right_contact = float(self.right_contact_var.get())
            self.modified_foot_contact[self.current_frame, 0] = left_contact
            self.modified_foot_contact[self.current_frame, 1] = right_contact
            self.contact_modified = True
            print(f"帧 {self.current_frame + 1} 接触状态已更新: 左脚={left_contact}, 右脚={right_contact}")
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字")
            self.load_contact_values()

    def save_to_file(self):
        """保存修改后的数据到文件"""
        if not self.contact_modified:
            messagebox.showinfo("提示", "接触状态未被修改，无需保存")
            return

        try:
            save_data = {}
            for key in self.npzdata.keys():
                if key == 'foot_contact':
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

    def go_to_frame(self):
        """跳转到指定帧"""
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
        if not self.is_playing:
            return

        if self.current_frame >= self.num_frames - 1:
            self.current_frame = 0

        self.load_frame(self.current_frame)
        self.current_frame += 1

        if self.is_playing and self.current_frame < self.num_frames:
            self.root.after(20, self.play_animation)  # 50Hz
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

    # 简化版的轨迹绘制功能
    def plot_joint_trajectories(self):
        """绘制关节轨迹"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 5, figsize=(15, 12))
        axes = axes.flatten()

        for i, joint_name in enumerate(self.joint_names[:25]):  # 最多显示25个关节
            if i >= len(axes):
                break

            joint_positions = self.qpos_23d[:, 7 + i]
            ax = axes[i]
            ax.plot(joint_positions)
            ax.set_title(joint_name)
            ax.axvline(x=self.current_frame, color='r', linestyle='--', alpha=0.7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run(self):
        """运行分析器"""
        self.root.mainloop()
        if self.viewer:
            self.viewer.close()


if __name__ == "__main__":
    path = "data/mocap/example.npz"
    analyzer = NPZAnalyzer(path)
    analyzer.run()
