# pkl&npz_conversion 工具集
这个文件夹包含了一组用于处理机器人运动数据格式的脚本，主要支持 PKL 和 NPZ 格式之间的相互转换，以及数据分析和可视化。

## 脚本概述
### convert_npz_to_pickle.py
将 NPZ 格式的运动数据转换为 PKL 格式。

功能特点：

读取包含 qpos 数据的 NPZ 文件
处理 36 维的 qpos 数据（3维根位置 + 4维根方向四元数 + 29维自由度）
计算相对于根坐标系的局部身体位置
支持机器人的层级链接结构
保存转换后的数据为 PKL 格式
支持批量转换，当传入值为文件夹时，转换这个文件夹中的所有npz；当传入值为npz文件时，只转换当前传入的npz

使用方法：
```bash
python convert_npz_to_pickle.py <包含npz的文件夹路径或npz文件路径>
```


### convert_pickle_to_npz.py
将 PKL 格式的运动数据转换为 NPZ 格式。

功能特点：

读取 PKL 文件中的运动数据
将四元数从 XYZW 格式转换为 WXYZ 格式
拼接根位置、根方向和自由度数据
导出为兼容 MuJoCo 的 NPZ 格式
支持批量转换，当传入值为文件夹时，转换这个文件夹中的所有pkl；当传入值为pkl文件时，只转换当前传入的pkl

使用方法：

```bash
python convert_pickle_to_npz.py <包含pkl的文件夹路径>
```

### check_and_export_pickle.py
PKL 文件分析工具，提供可视化和超限检测功能。

功能特点：

可视化机器人运动数据
检测关节位置和速度超限
支持多种机器人模型（66155 和 66377 模式）
支持播放动画和帧控制
可以批量浏览多个 PKL 文件
支持将 PKL 数据导出为 NPZ 格式
使用方法：

```bash
python check_and_export_pickle.py
```
注意：此脚本需要相应的 XML 模型文件来可视化机器人。

### read_pickle.py
PKL 文件读取和分析工具。

功能特点：

读取和分析 PKL 文件结构
显示数据的基本统计信息
支持批量分析多个文件
显示数组形状、数据类型等元数据
使用方法：

```bash
python read_pickle.py
```
注意：此脚本需要手动修改PKL文件路径。

## 数据格式说明
PKL 格式（用于TWIST2训练）
PKL 文件包含以下字段：

fps : 动画帧率
root_pos : 根位置 (N×3)
root_rot : 根方向四元数 (N×4, XYZW格式)
dof_pos : 自由度位置 (N×23 或 N×29)
local_body_pos : 局部身体位置 (N×38×3)
link_body_list : 链接名称列表


NPZ 格式（用于Galbot_mj训练）
NPZ 文件包含以下字段：

qpos : 关节位置 (N×30 或 N×36, WXYZ四元数格式)
frequence : 动画频率
机器人模型支持
脚本支持以下机器人模型：

66155 模式：23 自由度的机器人模型
66377 模式：29 自由度的机器人模型


## 依赖项
numpy
scipy
mujoco (用于可视化)
tkinter (用于 GUI)

## 注意事项
四元数格式转换：PKL 文件使用 XYZW 格式，而 NPZ 文件使用 WXYZ 格式
路径配置：某些脚本可能需要根据您的项目结构调整路径
模型文件：check_and_export_pickle.py 需要正确的 XML 模型文件进行可视化
数据兼容性：转换脚本假设特定的数据结构，可能需要根据实际情况调整
文件结构
pkl&npz_conversion/
├── convert_npz_to_pickle.py    # NPZ转PKL
├── convert_pickle_to_npz.py    # PKL转NPZ
├── check_and_export_pickle.py   # PKL分析和可视化工具
├── read_pickle.py          # PKL读取和分析工具
└── README.md            # 本文件