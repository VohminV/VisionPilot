from ultralytics import YOLO

model = YOLO("Y.pt")
model.export(format="onnx", task="segment")
