from ultralytics import YOLO

if __name__ == '__main__':
    # 加载官方大模型
    model = YOLO("yolov8n.pt")

    # 启动训练
    print("🚀 炼丹炉正在启动，准备让显卡咆哮...")
    model.train(data="snack.yaml", epochs=50, imgsz=640, batch=8)