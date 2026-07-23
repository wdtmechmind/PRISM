# 硬件触发快速启动

## ⚡ 一句话

PRISM 已迁移至硬件触发同步模式。4 个相机在硬件级别严格同步（< 1 ms）。

---

## 📋 首次设置（一次性）

### 1. 标定相机

```bash
python3 tools/prism_charuco_calibration_capture.py \
  --output-dir ~/mvs_charuco_data \
  --squares-x 12 --squares-y 9 \
  --square-length-mm 15 --marker-length-mm 11.25
```

详见 [标定指南](HARDWARE_TRIGGER_CALIBRATION_GUIDE.md)

### 2. 运行官方标定

```bash
python3 /opt/MVS/Samples/64/Python/General/Recording/CharucoCalibrate4Cam.py \
  --dataset-root ~/mvs_charuco_data/CharucoCapture_[时间戳] \
  --squares-x 12 --squares-y 9 \
  --square-length-mm 15 --marker-length-mm 11.25 \
  --aruco-dict DICT_5X5_1000 \
  --output ~/mvs_charuco_data/charuco_4cam_result.json
```

### 3. 更新配置

```bash
cp ~/mvs_charuco_data/charuco_4cam_result.json \
   /mnt/projects-8tb/PRISM/configs/devices/charuco_4cam_result.json
```

---

## 🚀 启动采集

```bash
scripts/collect_task.sh \
  --task-name grasp-demo \
  --num-trials 20 \
  --output-dir data/raw
```

**自动处理**：
- ✓ 识别 Master 相机 (DA8165486)
- ✓ 配置 Slave 相机 (cam1-3)
- ✓ GPIO 硬件触发自动启用

---

## ⚙️ 硬件连接（标定前必须检查）

| 接口 | 用途 | 方向 |
|------|-----|------|
| GPIO Line0 | 触发输入 | ← cam1-3 |
| GPIO Line1 | 触发输出 | → DA8165486 |

---

## 📚 详细文档

- [完整标定指南](HARDWARE_TRIGGER_CALIBRATION_GUIDE.md) — 步骤详解、故障排查
- [项目主文档](../README.md#5-多相机硬件触发同步) — 采集参数和性能对比

## ❓ 常见问题

**Q: 我需要更新现有代码吗？**  
A: 如果直接使用 `open_and_prepare()`，需要更新参数。session_manager 已自动处理。

**Q: 旧的标定数据能用吗？**  
A: 建议重新标定。硬件触发的同步精度不同，可能影响标定结果的准确度。

**Q: 能否回到自由运行模式？**  
A: 当前版本已移除自由运行支持。若需要，可从 git 历史恢复旧代码。

**Q: GPIO 接线出错会怎样？**  
A: 从相机会一直等待触发信号，最终超时。检查 Line0/Line1 连接。

---

**改造日期**: 2026-07-23  
**维护者**: PRISM Team
