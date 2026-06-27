from ultralytics import YOLO

# 1. 加载你亲手炼制的专属大模型（注意这里的相对路径要写对）
model = YOLO("runs/detect/train/weights/best.pt")

# 2. 调起摄像头，实时无情锁定魔芋爽！
model.predict(source="0", show=True)