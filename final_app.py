"""
final_app.py — 工业级 AI 零食检测系统（多线程 + 配置驱动 + 日志记录）

架构：
┌──────────────────────────────┐    ┌──────────────────────────────┐
│         Worker 线程           │    │          主线程 (GUI)         │
│                              │    │                              │
│  QTimer (fps ms)             │    │  on_frame_ready(QImage,int)  │
│    ↓                         │    │    ├── QLabel.setPixmap()    │
│  cv2.VideoCapture.read()     │    │    └── QLabel.setText()      │
│    ↓                         │    │                              │
│  YOLO.model.predict()        │    │  on_error(str)               │
│    ↓                         │    │    └── 显示错误信息           │
│  results[0].plot()           │    │                              │
│    ↓                         │    │  toggle_camera()             │
│  BGR→RGB→QImage.copy()      │    │    └── 启动/停止线程          │
│    ↓                         │    │                              │
│  Signal(frame_ready) ────────→    │                              │
└──────────────────────────────┘    └──────────────────────────────┘
"""

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import yaml
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

# ============================================================================
#  日志系统初始化
# ============================================================================

LOG = logging.getLogger("VisionApp")  # 根 logger，后续由 setup_logging 配置


def setup_logging(config: dict) -> None:
    """根据配置字典初始化全局日志系统（同时输出到文件和控制台）"""
    log_cfg = config.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s")
    datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    # 根 logger 设置
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有的 handler（防止重复添加）
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # 文件 handler
    log_file = log_cfg.get("file", "app.log")
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
    except Exception as e:
        # 文件日志创建失败至少还能在控制台看到
        print(f"⚠ 无法创建日志文件 {log_file}: {e}", file=sys.stderr)

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # 更新模块级 logger
    global LOG
    LOG = logging.getLogger("VisionApp")
    LOG.info("日志系统初始化完成 (level=%s, file=%s)", level_str, log_file)


# ============================================================================
#  配置文件加载
# ============================================================================

# 当 config.yaml 缺失时使用的出厂默认值（保证程序至少能启动）
_DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "name": "AI 零食检测系统",
        "window_width": 800,
        "window_height": 600,
    },
    "camera": {
        "index": 0,
        "fps": 30,
    },
    "model": {
        "path": "runs/detect/train/weights/best.pt",
        "conf_threshold": 0.8,
    },
    "logging": {
        "level": "INFO",
        "file": "app.log",
        "format": "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    },
    "ui": {
        "snack_label": "魔芋爽",
        "plentiful_threshold": 2,
        "messages": {
            "ready": "🟢 系统准备就绪",
            "plentiful": "🟢 零食库存充足，当前数量：{count}",
            "low": "⚠️ 警告：魔芋爽仅剩最后一包，请省着点吃！",
            "empty": "🚨 警报：画面中未检测到魔芋爽！立刻下楼去买！",
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个字典，override 中的值优先"""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str = "config.yaml") -> dict:
    """
    加载 YAML 配置文件并合并默认值。
    无论发生什么错误都不会崩溃，始终返回可用配置。
    """
    # 先获取 logger（此时可能尚未配置 handler，但 logging 会使用 lastResort）
    logger = logging.getLogger("VisionApp")

    config = dict(_DEFAULT_CONFIG)  # shallow copy of top level

    path = Path(config_path)
    if not path.exists():
        logger.warning(
            "配置文件 %s 不存在，将使用默认配置。请创建 config.yaml 以自定义参数。",
            path.absolute(),
        )
        return config

    try:
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error("配置文件 %s 解析失败: %s。将使用默认配置。", path.absolute(), e)
        return config
    except OSError as e:
        logger.error("无法读取配置文件 %s: %s。将使用默认配置。", path.absolute(), e)
        return config

    if user_config is None:
        logger.warning("配置文件 %s 为空，将使用默认配置。", path.absolute())
        return config

    if not isinstance(user_config, dict):
        logger.error("配置文件格式错误（应为字典），将使用默认配置。")
        return config

    config = _deep_merge(config, user_config)
    logger.info("配置文件加载成功: %s", path.absolute())
    return config


# ============================================================================
#  Worker 线程 — 专司视频捕获 + AI 推理，绝不触碰 UI
# ============================================================================

class DetectionWorker(QObject):
    """
    独立 Worker：运行在子线程中，负责：
    1. 打开摄像头并循环读取视频帧
    2. 调用 YOLO 模型进行推理
    3. 在图像上绘制检测框
    4. 通过 Signal 将处理结果异步发射回主线程

    所有参数由调用方注入（无硬编码依赖）。
    """

    # ---- 信号定义 ----
    frame_ready = Signal(QImage, int)    # (已标注画面, 检测到的零食数量)
    error_occurred = Signal(str)         # 错误消息

    def __init__(
        self,
        model_path: str,
        camera_index: int,
        conf_threshold: float,
        fps_interval: int,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.model_path = model_path
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.fps_interval = fps_interval

        # 这些资源在子线程中初始化（见 start_capture）
        self.model: YOLO | None = None
        self.cap: cv2.VideoCapture | None = None

        # 帧读取连续失败计数器（用于检测摄像头断开）
        self._consecutive_failures: int = 0
        self._max_failures_before_warn: int = 30  # ~1 秒（30 fps 下）

        # Worker 自带的定时器：必须传入 self 作为 parent，
        # 这样 moveToThread 时定时器才会跟随 Worker 一起迁移到子线程
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._process_frame)

        self._active = False

        self._logger = logging.getLogger("VisionApp.Worker")

    # ---- 公开槽函数（由主线程通过信号/跨线程调用触发） ----

    @Slot()
    def start_capture(self):
        """在 Worker 线程中初始化模型与摄像头，然后启动定时器"""
        self._logger.info("Worker 线程启动中...")

        # 1. 加载 YOLO 模型
        try:
            self._logger.info("正在加载 YOLO 模型: %s", self.model_path)
            self.model = YOLO(self.model_path)
            self._logger.info("YOLO 模型加载成功")
        except FileNotFoundError as e:
            msg = f"模型文件不存在: {self.model_path}"
            self._logger.error(msg)
            self.error_occurred.emit(f"模型加载失败：{msg}")
            return
        except Exception:
            msg = f"模型加载异常: {traceback.format_exc()}"
            self._logger.error(msg)
            self.error_occurred.emit("模型加载失败，请查看 app.log 了解详情")
            return

        # 2. 打开摄像头
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
        except Exception:
            self._logger.error("创建 VideoCapture 时异常:\n%s", traceback.format_exc())
            self.error_occurred.emit(f"无法访问摄像头 (索引 {self.camera_index})")
            return

        if not self.cap.isOpened():
            self._logger.error("无法打开摄像头 (索引 %d)", self.camera_index)
            self.error_occurred.emit(f"无法打开摄像头 (索引 {self.camera_index})")
            self.cap = None
            return

        self._logger.info(
            "摄像头已打开 (索引=%d, 分辨率=%dx%d)",
            self.camera_index,
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )

        # 3. 启动定时器
        self._consecutive_failures = 0
        self._active = True
        self._timer.start(self.fps_interval)
        self._logger.info("检测循环已启动 (间隔=%dms, conf=%.2f)", self.fps_interval, self.conf_threshold)

    @Slot()
    def stop_capture(self):
        """停止捕获循环，释放硬件资源"""
        self._logger.info("Worker 线程停止中...")
        self._active = False
        self._timer.stop()

        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self._logger.info("摄像头已释放")

        self.model = None

    # ---- 核心循环（仅在 Worker 线程中执行） ----

    def _process_frame(self):
        """读取一帧 → YOLO 推理 → 画框 → 发射信号"""
        if not self._active or self.cap is None:
            return

        try:
            ret, frame = self.cap.read()
        except Exception:
            self._logger.error("cv2.VideoCapture.read() 异常:\n%s", traceback.format_exc())
            self.error_occurred.emit("摄像头读取异常，请检查硬件连接")
            self._active = False
            self._timer.stop()
            return

        if not ret or frame is None:
            self._consecutive_failures += 1
            if self._consecutive_failures == 1:
                self._logger.warning("摄像头帧读取失败（可能暂时断开）")
            elif self._consecutive_failures >= self._max_failures_before_warn:
                self._logger.error(
                    "摄像头连续 %d 帧读取失败，设备可能已断开", self._consecutive_failures
                )
                self.error_occurred.emit("摄像头连接断开！请检查 USB 线缆")
                self._active = False
                self._timer.stop()
            return
        else:
            self._consecutive_failures = 0  # 复位

        # --- YOLO 推理 ---
        try:
            results = self.model.predict(
                source=frame,
                verbose=False,
                conf=self.conf_threshold,
            )
            snack_count = len(results[0].boxes)
        except Exception:
            self._logger.error("YOLO 推理异常:\n%s", traceback.format_exc())
            self.error_occurred.emit("AI 推理出错，请查看 app.log")
            return

        # --- 绘制检测框 ---
        try:
            annotated = results[0].plot()
        except Exception:
            self._logger.error("绘制检测框异常:\n%s", traceback.format_exc())
            # 即使画框失败也尝试继续（直接使用原始 frame）
            annotated = frame

        # --- 颜色空间转换：BGR → RGB ---
        try:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        except Exception:
            self._logger.error("颜色转换异常:\n%s", traceback.format_exc())
            return

        # --- 构造 QImage 并立即 .copy() 脱离 numpy 内存 ---
        try:
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        except Exception:
            self._logger.error("QImage 构造异常:\n%s", traceback.format_exc())
            return

        # --- 发射信号，将数据交给主线程渲染 ---
        self.frame_ready.emit(qt_img, snack_count)


# ============================================================================
#  主窗口 — 只负责 UI 布局与渲染，所有重计算已剥离到 Worker 线程
# ============================================================================

class VisionApp(QMainWindow):
    """主窗口：仅持有 UI 控件，响应 Worker 发来的信号更新画面"""

    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._app_cfg = config.get("app", {})
        self._ui_cfg = config.get("ui", {})
        self._camera_cfg = config.get("camera", {})
        self._model_cfg = config.get("model", {})

        self._logger = logging.getLogger("VisionApp.GUI")

        # ---- 窗口基本属性 ----
        self.setWindowTitle(self._app_cfg.get("name", "AI 零食检测系统"))
        self.resize(
            self._app_cfg.get("window_width", 800),
            self._app_cfg.get("window_height", 600),
        )

        # ---- 提取 UI 文案 ----
        msgs = self._ui_cfg.get("messages", {})
        self._msg_ready = msgs.get("ready", "🟢 系统准备就绪")
        self._msg_plentiful = msgs.get("plentiful", "🟢 零食库存充足，当前数量：{count}")
        self._msg_low = msgs.get("low", "⚠️ 警告：仅剩最后一包！")
        self._msg_empty = msgs.get("empty", "🚨 警报：未检测到物品！")
        self._threshold = self._ui_cfg.get("plentiful_threshold", 2)

        # ================ UI 搭建 ================
        self._build_ui()

        # ================ 多线程基础设施 ================
        self._worker_thread: QThread | None = None
        self._worker: DetectionWorker | None = None

        self._logger.info("主窗口初始化完成")

    # ---- UI 构建 ----

    def _build_ui(self):
        """搭建界面控件（纯布局，不涉及业务逻辑）"""
        self.video_label = QLabel("点击下方按钮启动系统...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: black; color: white; font-size: 20px;"
        )
        self.video_label.setScaledContents(True)
        self.video_label.setMinimumSize(640, 480)

        self.count_label = QLabel(self._msg_ready)
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #2ECC71;"
        )
        self.count_label.setMinimumHeight(40)

        self.btn_open = QPushButton("🔥 打开摄像头并启动 AI")
        self.btn_open.setMinimumHeight(50)
        self.btn_open.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.btn_open.clicked.connect(self._toggle_camera)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.count_label)
        layout.addWidget(self.btn_open)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # ---- 线程管理 ----

    def _create_worker(self):
        """创建 Worker + QThread 并连接信号（每次启动时调用）"""
        self._logger.info("创建 DetectionWorker...")

        self._worker = DetectionWorker(
            model_path=self._model_cfg.get("path", "runs/detect/train/weights/best.pt"),
            camera_index=self._camera_cfg.get("index", 0),
            conf_threshold=float(self._model_cfg.get("conf_threshold", 0.8)),
            fps_interval=self._camera_cfg.get("fps", 30),
        )
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)

        # ---- 信号接线 ----
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.error_occurred.connect(self._on_error)
        self._worker_thread.started.connect(self._worker.start_capture)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

    def _toggle_camera(self):
        """启动 / 停止检测系统"""
        if self._worker_thread is not None and self._worker_thread.isRunning():
            # ===== 停止 =====
            self._logger.info("用户请求停止检测")
            self._worker.stop_capture()
            self._worker_thread.quit()
            self._worker_thread.wait()

            self._worker = None
            self._worker_thread = None

            self.btn_open.setText("🔥 打开摄像头并启动 AI")
            self.btn_open.setEnabled(True)
            self.video_label.clear()
            self.video_label.setText("点击下方按钮启动系统...")
            self.count_label.setText(self._msg_ready)
            self.count_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: #2ECC71;"
            )
        else:
            # ===== 启动 =====
            self._logger.info("用户请求启动检测")
            self._create_worker()
            self._worker_thread.start()

            self.btn_open.setText("⚙️ AI 视觉系统实时监控中...")
            self.btn_open.setEnabled(False)

    # ---- UI 更新槽函数（仅主线程调用） ----

    @Slot(QImage, int)
    def _on_frame_ready(self, qt_img: QImage, snack_count: int):
        """接收 Worker 发来的帧和计数，仅做 UI 渲染"""
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

        if snack_count >= self._threshold:
            text = self._msg_plentiful.format(count=snack_count)
            color = "#2ECC71"
        elif snack_count == 1:
            text = self._msg_low
            color = "#F39C12"
        else:
            text = self._msg_empty
            color = "#E74C3C"

        self.count_label.setText(text)
        self.count_label.setStyleSheet(
            f"font-size: 24px; color: {color}; font-weight: bold;"
        )

    @Slot(str)
    def _on_error(self, message: str):
        """Worker 线程报错时更新 UI"""
        self._logger.error("Worker 错误报告: %s", message)
        self.count_label.setText(f"❌ {message}")
        self.count_label.setStyleSheet(
            "font-size: 24px; color: #E74C3C; font-weight: bold;"
        )
        self.btn_open.setText("🔥 打开摄像头并启动 AI")
        self.btn_open.setEnabled(True)

    # ---- 生命周期 ----

    def closeEvent(self, event):
        """窗口关闭时安全停止子线程并释放硬件"""
        self._logger.info("应用程序正在退出...")
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker.stop_capture()
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._logger.info("Worker 线程已安全退出")
        event.accept()


# ============================================================================
#  应用入口
# ============================================================================

def main() -> int:
    """
    主函数：加载配置 → 初始化日志 → 启动 GUI。
    任何早期错误都会被捕获并记录，不会直接崩溃。
    """
    # 1. 加载配置文件（文件缺失也能继续运行）
    try:
        config = load_config("config.yaml")
    except Exception:
        # 极极端情况：load_config 本身抛异常（理论上不会，但防御性编程）
        print("严重错误：配置加载模块自身崩溃，使用完全默认配置。", file=sys.stderr)
        traceback.print_exc()
        config = _DEFAULT_CONFIG

    # 2. 初始化日志系统
    try:
        setup_logging(config)
    except Exception:
        print("严重错误：日志系统初始化失败，使用默认日志配置。", file=sys.stderr)
        traceback.print_exc()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-8s] %(name)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    LOG.info("=" * 50)
    LOG.info("%s 启动", config.get("app", {}).get("name", "AI 零食检测系统"))
    LOG.info("=" * 50)

    # 3. 启动 Qt 应用
    try:
        app = QApplication(sys.argv)
        window = VisionApp(config)
        window.show()
        exit_code = app.exec()
    except Exception:
        LOG.critical("应用运行时发生无法恢复的错误:\n%s", traceback.format_exc())
        return 1

    LOG.info("应用正常退出 (exit_code=%d)", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
