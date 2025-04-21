import torch
import sys

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
    print("cuDNN版本:", torch.backends.cudnn.version())
    print("当前显卡:", torch.cuda.get_device_name(0))
    print("显卡数量:", torch.cuda.device_count())
    
    # 测试CUDA张量
    print("\nCUDA张量测试:")
    cuda_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    print("CUDA张量:", cuda_tensor)
    print("CUDA张量设备:", cuda_tensor.device)
else:
    print("警告: CUDA不可用，将使用CPU进行计算")

# 测试基本张量操作
print("\n基本张量操作:")
x = torch.tensor([[1, -1], [2, 3]])
print("创建张量:\n", x)
x = torch.zeros([2, 2])
print("零张量:\n", x)
x = torch.ones([2, 2, 3])
print("单位张量:\n", x)
