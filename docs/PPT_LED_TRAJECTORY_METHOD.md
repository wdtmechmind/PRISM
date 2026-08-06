# 根据 LED 重建轨迹的方法（技术版 PPT 文案）

## 第 1 页：任务与输入输出

标题：基于多相机 LED 的三维轨迹与 6D 位姿恢复

文案：
- 输入：4 路硬触发同步相机图像、ChArUco 标定文件、LED 颜色先验（R/Y/B/G）。
- 输出：
        - 每个颜色 LED 的 3D 轨迹 $\mathbf{p}_c(t)$
        - 刚体位姿 $\{\mathbf{R}(t),\mathbf{t}(t)\}$ 与 RPY
        - 轨迹 CSV、位姿 CSV、质量报告
- 运行模式：在线重建（实时）+ 离线重建（可复现）。

---

## 第 2 页：最低观测条件（你关心的“需要几个视野、几个 LED”）

标题：可解条件与退化条件

文案：
- 单个颜色 LED 的三角化条件：
        - 同一时刻至少 2 个相机看到该颜色点。
        - 少于 2 视角时，该颜色本帧不可三角化。
- 刚体位姿的估计条件：
        - 至少 3 个非共线 LED 的 3D 点同时可用。
        - 少于 3 个点无法解 6D 位姿（代码直接返回空）。
- 实践建议：
        - 为稳定性，尽量保证每帧 3 到 4 个 LED 都可见。
        - 在遮挡场景下，允许短时丢点，由 Kalman 预测补偿。

讲解词：
一句话：三角化看的是“每个颜色至少 2 视角”，姿态看的是“同一帧至少 3 个 LED 3D 点”。

---

## 第 3 页：用了哪些标定参数（内参/外参/坐标系）

标题：相机模型与坐标系定义

文案：
- 标定来源：`charuco_4cam_result.json`。
- 每个相机使用：
        - 内参 $\mathbf{K}_i$（焦距、主点）
        - 畸变参数 $\mathbf{D}_i$
        - 外参 $\mathbf{R}_i, \mathbf{t}_i$
- 实现细节：
        - 先用 $\mathbf{K}_i,\mathbf{D}_i$ 做去畸变，得到归一化像平面点。
        - 用投影矩阵 $\mathbf{P}_i=[\mathbf{R}_i|\mathbf{t}_i]$ 建立线性三角化方程。
- 当前工程的 3D 点默认在 cam0 参考系下（cam0: $\mathbf{R}=\mathbf{I},\mathbf{t}=\mathbf{0}$）。
- corrected frame（校正坐标系）主要用于可视化/分析对齐，不改变三角化核心求解。

---

## 第 4 页：总体流程图（配图 1）

标题：方法流程总览

                                                 同步多相机图像（硬触发）
                                                                                                |
                                                                                                v
                                                                LED 2D 检测（HSV / YOLO / Hybrid）
                                                                                                |
                                                                                                v
                                 按颜色聚合跨视角观测 { (u,v)_i }_{i=1...N}
                                                                                                |
                                                                                                v
                                                         多视角三角化 -> 3D 点 p_c(t)
                                                                                                |
                                                                                                v
                                        鲁棒筛选（误差阈值 + 异常视角剔除）
                                                                                                |
                                                                                                v
                                                        卡尔曼滤波与短时预测补点
                                                                                                |
                                                                                                v
                                                刚体位姿估计（SVD / Kabsch）
                                                                                                |
                                                                                                v
                                        导出轨迹 CSV + 位姿 CSV + 质量报告

---

## 第 5 页：二维检测模型

标题：从像素空间提取 LED 观测

文案：
- HSV：颜色阈值 + 形态学 + 轮廓质心。
- YOLO：框中心点映射到颜色类别。
- Hybrid：优先 YOLO，缺失时回退 HSV。
- 观测定义：$\mathbf{z}_{i,c}(t)=(u_{i,c}(t),v_{i,c}(t))$。

---

## 第 6 页：三角化公式（完整可讲）

标题：从 2D 到 3D 的计算公式

文案：
- 去畸变后得到归一化观测 $\tilde{\mathbf{u}}_i=(\tilde u_i,\tilde v_i)$。
- 对每个有效相机 $i$，设 $\mathbf{P}_i$ 的第 1/2/3 行为 $\mathbf{p}_{i1},\mathbf{p}_{i2},\mathbf{p}_{i3}$，构造：

$$
	ilde u_i\,\mathbf{p}_{i3}^\top-\mathbf{p}_{i1}^\top=0,
\quad
	ilde v_i\,\mathbf{p}_{i3}^\top-\mathbf{p}_{i2}^\top=0
$$

- 叠加所有视角得线性系统 $\mathbf{A}\mathbf{X}_h=0$，解：

$$
\mathbf{X}_h^*=\arg\min_{\|\mathbf{X}_h\|=1}\|\mathbf{A}\mathbf{X}_h\|,
\quad
\mathbf{X}=\left[\frac{X}{W},\frac{Y}{W},\frac{Z}{W}\right]^\top
$$

- 重投影误差门控：

$$
e_i=\left\|\pi(\mathbf{R}_i\mathbf{X}+\mathbf{t}_i)-\tilde{\mathbf{u}}_i\right\|_2
$$

- 若 $\max_i e_i>\tau$，尝试剔除 1 个异常视角后重算；仍超阈值则拒绝该点。

---

## 第 7 页：时序滤波与丢点策略

标题：轨迹连续性增强

文案：
- 每个颜色独立卡尔曼状态：

$$
\mathbf{x}_k=[x,y,z,v_x,v_y,v_z]^\top
$$

$$
\mathbf{x}_k=\mathbf{F}(\Delta t)\mathbf{x}_{k-1}+\mathbf{w}_k,
\quad
\mathbf{z}_k=\mathbf{H}\mathbf{x}_k+\mathbf{v}_k
$$

- 测量存在：`measured`（三角化 + 校正）。
- 短时缺失：`predicted`（仅预测补点）。
- 连续缺失超阈值：`lost`（重置该颜色滤波器）。

---

## 第 8 页：LED 到刚体 6D 位姿

标题：SVD/Kabsch 配准

文案：
- 模型点 $\{\mathbf{q}_j\}$ 来自初始化时刻的 LED 布局。
- 当前观测点 $\{\mathbf{p}_j\}$。
- 目标函数：

$$
\min_{\mathbf{R},\mathbf{t}}\sum_j\|\mathbf{p}_j-(\mathbf{R}\mathbf{q}_j+\mathbf{t})\|^2,
\quad \mathbf{R}\in SO(3)
$$

- 由 SVD 解旋转，再由质心关系求平移：

$$
\mathbf{t}=\bar{\mathbf{p}}-\mathbf{R}\bar{\mathbf{q}}
$$

- 输出 roll/pitch/yaw + translation 时间序列。

---

## 第 9 页：输出与评价

标题：结果文件与指标

文案：
- 输出：LED 轨迹 CSV、刚体位姿 CSV、质量报告（md/json）。
- 指标：最大重投影误差、有效跟踪率、丢点段长度、在线离线一致性。

---

## 第 10 页：一句话总结

文案：
方法核心是“2D 检测 + 多视角几何 + 时序滤波 + 刚体配准”：在明确可解条件下，用相机内外参把多视角像素观测转成稳定的 3D 轨迹和 6D 位姿。