# Experiment-2：旋转与变换（MVP）

本项目是计算机图形学课程实验二的实现，使用 Taichi 框架完成了一个三维线框三角形的 MVP（Model-View-Projection）变换与交互旋转演示。程序运行后会弹出一个 `700x700` 的窗口，按下 `A` / `D` 键可控制图形绕 `Z` 轴旋转，按 `Esc` 退出。

## 一、项目结构

```
CG-lab/
├── main.py                # 根目录入口（预留）
├── pyproject.toml         # 项目配置与依赖声明
├── uv.lock                # 依赖版本锁定
└── src/
    ├── Work0/             # 实验 0
    └── work1/             # 实验 2（旋转与变换）
        ├── __init__.py
        ├── config.py      # 实验参数配置（窗口、相机、裁剪面、按键步长）
        └── main.py        # MVP 矩阵实现与渲染主循环
```

项目采用 `src` 布局，`work1` 为本次实验包，内部按职责拆分为两个核心模块：

- **config.py**：集中存放窗口参数、相机参数、投影参数和图元配置，如 `EYE_FOV`、`Z_NEAR`、`Z_FAR`、`TRIANGLE_VERTICES` 等。
- **main.py**：实现 `Model / View / Projection` 三个矩阵函数，完成顶点的齐次坐标变换、透视除法和屏幕映射，并驱动 GUI 渲染循环。

## 二、实现思路

### 2.1 模型变换（Model）

`get_model_matrix(angle)` 接收角度（角度制），先转换为弧度，再构造绕 `Z` 轴旋转矩阵：

```
[ cosθ  -sinθ  0  0 ]
[ sinθ   cosθ  0  0 ]
[  0      0    1  0 ]
[  0      0    0  1 ]
```

该矩阵用于实现图形在模型空间中的旋转效果。

### 2.2 视图变换（View）

`get_view_matrix(eye_pos)` 根据相机位置构造平移矩阵，将“相机平移到原点”等价为“场景按相反方向平移”：

```
[ 1  0  0  -eye_x ]
[ 0  1  0  -eye_y ]
[ 0  0  1  -eye_z ]
[ 0  0  0    1    ]
```

本实验中相机位置设为 `(0, 0, 5)`，观察方向沿 `-Z`。

### 2.3 投影变换（Projection）

`get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)` 按“透视到正交 + 正交投影”两步实现：

1. 先根据视场角和近裁剪面计算边界：
   - $t = \tan(\frac{fov}{2}) \cdot |n|$
   - $b = -t$
   - $r = aspect\_ratio \cdot t$
   - $l = -r$
2. 构造 $M_{persp\to ortho}$，将平截头体挤压到长方体。
3. 构造正交平移与缩放矩阵，得到 $M_{ortho}$。
4. 最终投影矩阵：$M_{proj} = M_{ortho} @ M_{persp\to ortho}$。

注意：本实验采用右手坐标系并看向 `-Z`，因此计算中使用 `n = -zNear`、`f = -zFar`。

### 2.4 MVP 合成与屏幕映射

对每个顶点使用列向量右乘规则：

$$MVP = M_{proj} @ M_{view} @ M_{model}$$

得到齐次坐标 `(x, y, z, w)` 后进行透视除法：

- `x_ndc = x / w`
- `y_ndc = y / w`
- `z_ndc = z / w`

再将 NDC 区间 `[-1, 1]` 映射到屏幕坐标区间 `[0, 1]`，用于 Taichi GUI 绘制线框三角形。

## 三、运行方法

环境要求：Python `>= 3.12`，Taichi `>= 1.7.4`，支持 CUDA/Vulkan 的 GPU（无独显时也可退化运行）。

```bash
git clone https://github.com/HoganPrice/experiment-1.git
cd experiment-1

# 安装依赖（使用 uv）
pip install uv
uv sync

# 运行实验二
uv run python -m src.work1.main
```

如果只想检查矩阵与投影结果，可使用 dry-run：

```bash
uv run python -m src.work1.main --dry-run
```

## 四、关键参数说明

以下参数定义在 `src/work1/config.py` 中：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `WINDOW_RES` | `(700, 700)` | 窗口分辨率 |
| `EYE_POS` | `(0.0, 0.0, 5.0)` | 相机位置 |
| `EYE_FOV` | `45.0` | 纵向视场角（角度制） |
| `ASPECT_RATIO` | `1.0` | 屏幕长宽比 |
| `Z_NEAR` | `0.1` | 近裁剪面距离 |
| `Z_FAR` | `50.0` | 远裁剪面距离 |
| `ROTATE_STEP_DEG` | `10.0` | 每次按键旋转步长（角度） |

## 五、交互与效果展示

程序成功运行后，窗口中可看到三条彩色边组成的线框三角形。按键交互如下：

- `A`：逆时针旋转（绕 `Z` 轴）
- `D`：顺时针旋转（绕 `Z` 轴）
- `Esc`：退出程序

可在本节补充你的运行截图/GIF（例如 `video.gif`）：

```markdown
![实验二运行效果](video.gif)
```

## 六、实验结论

通过本实验，完成了从三维模型坐标到二维屏幕坐标的完整变换链路，实现并验证了 `Model`、`View`、`Projection` 三个 `4x4` 齐次矩阵的构造方法，掌握了在 Taichi 中进行矩阵计算、齐次坐标透视除法和线框几何体实时渲染的基本流程。

## 七、依赖

- [Taichi](https://github.com/taichi-dev/taichi) >= 1.7.4
- Python >= 3.12
- 包管理工具：[uv](https://github.com/astral-sh/uv)
