import onnxruntime as ort
 
print(ort.get_device())
ort_session = ort.InferenceSession('/home/wdxm/code/YOLOs-CPP/models/yolo8n.onnx', providers=['CUDAExecutionProvider'])
print(ort_session.get_providers())