from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v12/yolov12-csf.yaml")
model.load("yolov12n.pt")
model.train(
    data="coco-csf.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    workers=4,
    device=0,
    lr0=0.001,
    lrf=0.01,
    project="runs/csf_coco",
    name="train",
    exist_ok=False,
    plots=True,
)
