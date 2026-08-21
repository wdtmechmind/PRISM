# Corrected Frame 定义

本文档说明 PRISM 如何根据四个 Hik 相机的 ChArUco 外参构造 `corrected frame`。

## 1. 坐标系关系

ChArUco 标定结果以 `cam0` 为参考相机。对第 `i` 个相机，外参满足：

$$
p_{cam_i} = R_{0\to i} p_{cam0} + t_{0\to i}
$$

因此四个相机中心首先都表达在 `cam0` 坐标系中：

$$
C_i = -R_i^T t_i
$$

`corrected frame` 是在这些相机中心的基础上构造的稳定坐标系。它不是预先测量的工作台坐标系。

## 2. 相机阵列平面

令四个相机中心为 $C_0,C_1,C_2,C_3$，相机阵列中心为：

$$
C_{center} = \frac{1}{4}\sum_{i=0}^{3} C_i
$$

代码对四个相机中心相对于 $C_{center}$ 的坐标做 SVD/PCA，取最小主方向作为相机阵列平面的法向量 $n$。由于实际相机安装不可能完全共面，这个平面是四个相机中心的最佳拟合平面。

## 3. corrected 原点

首先将四个相机中心投影到拟合平面上。

当四台相机都存在时，使用两条相机阵列对角线的交点作为原点：

- `cam0 -> cam2`
- `cam1 -> cam3`

它们在阵列平面内的交点定义为 $O_c$。如果相机不完整或无法求交，则退化使用相机中心平均值。

注意：$O_c$ 仍然是用 `cam0` 坐标表示的几何原点，不是工作台实际测量原点。

## 4. corrected Z 轴

每台相机在 `cam0` 世界中的光轴方向为：

$$
f_i = R_i^T
\begin{bmatrix}
0\\0\\1
\end{bmatrix}
$$

代码将四台相机的光轴方向求平均，得到 $f$，并利用它确定阵列平面法向量的正负方向。

最终选择法向量，使 corrected $+Z_c$ 方向与平均相机光轴方向相反。因此当前系统中通常有：

- 四个相机中心接近 $Z_c=0$ 平面；
- 被相机观察的工作空间位于 $Z_c<0$ 一侧；
- $+Z_c$ 大致指向远离相机的方向。

## 5. corrected X/Y 轴

将 `cam0 -> cam1` 的方向投影到相机阵列平面，并旋转到 corrected $+X_c$ 方向。因此：

- $+X_c$ 大致沿 `cam0 -> cam1`；
- $+Z_c$ 是相机阵列平面法向；
- $+Y_c$ 由右手坐标系确定。

当前四相机布局在 corrected frame 中大致为：

```text
cam3 -------- cam0
  |            |
  |            |
cam2 -------- cam1
```

## 6. 点坐标转换

对于 `cam0` 坐标系中的点 $p_0$，corrected 坐标为：

$$
p_c = R_c(p_0 - O_c)
$$

代码等价实现为：

```python
out = (points - origin) @ R.T
```

其中：

- `origin` 是 $O_c$；
- `R` 是从 `cam0` 轴方向旋转到 corrected frame 轴方向的矩阵；
- `out` 是 corrected frame 中的坐标。

本次转换使用的参数保存在：

[data/processed/realsense_assembly/task_20260805_163433_realsense_assembly/corrected_meta.json](../data/processed/realsense_assembly/task_20260805_163433_realsense_assembly/corrected_meta.json)

## 7. 刚体姿态转换

原始 LED 刚体旋转记为 $R_{0B}$，表示 LED rigid body frame $B$ 到 `cam0` frame 的姿态。corrected frame 中的旋转为：

$$
R_{cB} = R_c R_{0B}
$$

因此 `corrected_trajectory.csv` 中的：

- `x_m,y_m,z_m`：LED rigid body 原点在 corrected frame 中的位置；
- `qw,qx,qy,qz`：$R_{cB}$ 的四元数，顺序为 `w,x,y,z`；
- `roll_deg,pitch_deg,yaw_deg`：同一个 $R_{cB}$ 的 ZYX 欧拉角；
- `source_*`：转换前 `cam0` frame 中的原始位置和姿态。

## 8. 轨迹语义

当前 `corrected_trajectory.csv` 是固定 corrected frame 下的绝对 LED rigid body 轨迹，不是以其他动态参考系为锚点的相对轨迹。

## 9. 相关文件

- [Hik 四相机内外参](../configs/devices/charuco_4cam_result.json)
- [corrected frame 参数](../data/processed/realsense_assembly/task_20260805_163433_realsense_assembly/corrected_meta.json)
- [corrected 刚体轨迹](../data/processed/realsense_assembly/task_20260805_163433_realsense_assembly/corrected_trajectory.csv)
- [corrected 相机外参图](../data/processed/realsense_assembly/task_20260805_163433_realsense_assembly/corrected_camera_extrinsics.png)
- [实现代码](../src/prism/reconstruction/calibration.py)
