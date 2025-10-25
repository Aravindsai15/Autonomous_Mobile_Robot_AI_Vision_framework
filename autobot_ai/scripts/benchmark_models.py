import time
import numpy as np
import torch
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os




import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import os

# ------------------------------
# 2. Fast-SCNN Model Definition
# ------------------------------
class LearningToDownsample(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu6(self.bn3(self.conv3(x)))
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_ratio=6):
        super().__init__()
        hidden_channels = in_channels * expansion_ratio
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, 
                              stride=stride, padding=1, 
                              groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
#         self.se = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(out_channels, out_channels // 4, 1),  # Use out_channels here
#                 nn.ReLU(),
#                 nn.Conv2d(out_channels // 4, out_channels, 1),
#                 nn.Sigmoid()
#             )
        
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels//4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels//4, out_channels, 1),
            nn.Sigmoid()
        )
        
        
        self.stride = stride
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        if self.use_res_connect:
            se_weight = self.se(x)
            x = x * se_weight
            return x + identity
        return x

class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()
        layers = [
            Bottleneck(in_channels, 64, stride=2),
            Bottleneck(64, 64, stride=1),
            Bottleneck(64, 128, stride=2),
            Bottleneck(128, 128, stride=1),
            Bottleneck(128, out_channels, stride=1)
        ]
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)
        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class FastSCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.lds = LearningToDownsample()
        self.gfe = GlobalFeatureExtractor()
        self.ffm = FeatureFusionModule(in_channels=64+128)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        
        x1 = self.lds(x)
        x2 = self.gfe(x1)
        
        # Main branch
        x = self.ffm(x1, x2)
        main_out = self.classifier(x)
        main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=True)
        
        # Aux branch
        aux_out = self.aux_classifier(x2)
        aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=True)
        
        return main_out, aux_out





# Configuration
MODEL_DIR = r"/home/aravind/autobot_ai_ws/src/autobot_ai/models"  # Update this
INPUT_SHAPE = (1, 3, 512, 1024)  # Adjust to your model's input size
NUM_WARMUP = 10
NUM_TESTS = 100

def benchmark_pytorch(model_path):
    # 1. Load model class first (adjust import as needed)
    model = FastSCNN(num_classes=4).cuda()  # Initialize model
    
    # 2. Load weights properly
    state_dict = torch.load(model_path)
    if 'model' in state_dict:  # Handle nested state_dict
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Benchmark as before
    dummy_input = torch.randn(*INPUT_SHAPE).cuda()
    
    # Warmup
    for _ in range(NUM_WARMUP):
        _ = model(dummy_input)
    
    # Benchmark
    start = time.time()
    for _ in range(NUM_TESTS):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    latency = (time.time() - start) * 1000 / NUM_TESTS
    fps = 1000 / latency
    return latency, fps

def benchmark_onnx(model_path):
    sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    
    # Warmup
    for _ in range(NUM_WARMUP):
        _ = sess.run(None, {'input': dummy_input})
    
    # Benchmark
    start = time.time()
    for _ in range(NUM_TESTS):
        _ = sess.run(None, {'input': dummy_input})
    latency = (time.time() - start) * 1000 / NUM_TESTS
    fps = 1000 / latency
    return latency, fps

def benchmark_tensorrt(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    stream = cuda.Stream()
    
    # Allocate buffers
    dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    d_input = cuda.mem_alloc(dummy_input.nbytes)
    outputs = [np.empty(context.get_binding_shape(i), dtype=np.float32) for i in range(1, engine.num_bindings)]
    d_outputs = [cuda.mem_alloc(o.nbytes) for o in outputs]
    
    # Warmup
    for _ in range(NUM_WARMUP):
        cuda.memcpy_htod_async(d_input, dummy_input, stream)
        context.execute_async_v2(bindings=[int(d_input)] + [int(d) for d in d_outputs], stream_handle=stream.handle)
        stream.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(NUM_TESTS):
        cuda.memcpy_htod_async(d_input, dummy_input, stream)
        context.execute_async_v2(bindings=[int(d_input)] + [int(d) for d in d_outputs], stream_handle=stream.handle)
        stream.synchronize()
    latency = (time.time() - start) * 1000 / NUM_TESTS
    fps = 1000 / latency
    return latency, fps

if __name__ == "__main__":
    print(f"{'Model':<30} | {'Latency (ms)':>12} | {'FPS':>8}")
    print("-" * 60)
    
    for model_file in os.listdir(MODEL_DIR):
        model_path = os.path.join(MODEL_DIR, model_file)
        if model_file.endswith(".pth"):
            latency, fps = benchmark_pytorch(model_path)
        elif model_file.endswith(".onnx"):
            latency, fps = benchmark_onnx(model_path)
        elif model_file.endswith(".trt"):
            latency, fps = benchmark_tensorrt(model_path)
        else:
            continue
            
        print(f"{model_file:<30} | {latency:>12.2f} | {fps:>8.2f}")