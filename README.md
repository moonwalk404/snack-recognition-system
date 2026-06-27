# Snack Recognition System 零食识别系统

基于 **YOLOv8 + PySide6** 的工业级 AI 零食库存检测系统。通过摄像头实时监测桌面上魔芋爽的数量，并在库存不足时自动告警。

## 功能

- **实时目标检测** — YOLOv8 模型实时识别画面中的零食，绘制检测框与置信度
- **库存监控告警** — 三级库存状态：充足 🟢 / 告急 ⚠️ / 缺货 🚨
- **工业级架构** — 多线程采集+推理，GUI 不掉帧；YAML 配置驱动，无需改代码
- **日志系统** — 自动记录运行日志到文件，便于排查问题

## 项目结构

`
├── final_app.py          # 主程序：多线程 GUI + YOLO 推理
├── main.py               # 简化版上位机入口（单线程）
├── train_snack.py        # YOLOv8 训练脚本
├── test_snack.py         # 模型验证脚本
├── snack.yaml            # 数据集配置（类别: snack）
├── config.yaml           # 应用配置文件（摄像头、模型路径、阈值、告警文案等）
├── yolov8n.pt            # YOLOv8n 预训练权重
├── runs/detect/train/    # 训练结果：混淆矩阵、PR曲线、best.pt、last.pt
└── app.log               # 运行日志
`

## 快速启动

### 环境要求

- Python 3.10+
- 摄像头（USB 或笔记本内置）

### 安装依赖

`ash
pip install ultralytics opencv-python pyside6 pyyaml
`

### 运行

`ash
python final_app.py
`

## 配置说明

编辑 [config.yaml](config.yaml) 修改运行参数：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| pp.name | 窗口标题 | 专属 AI 零食检测系统 |
| camera.index | 摄像头设备索引 | 0 |
| camera.fps | 采集间隔（ms） | 30 |
| model.path | 模型权重路径 | runs/detect/train/weights/best.pt |
| model.conf_threshold | 置信度阈值 | 0.8 |
| ui.snack_label | 零食名称 | 魔芋爽 |
| ui.plentiful_threshold | 库存充足阈值 | 2 |
