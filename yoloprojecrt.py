from ultralytics import YOLO

# 1. 自动下载官方最轻量、速度最快的预训练模型（负责认出日常的80种东西）
model = YOLO("yolov8n.pt")

# 2. 让 AI 直接调用你的摄像头进行实时检测，并把结果弹窗显示出来
model.predict(source="0", show=True)