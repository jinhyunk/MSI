import torch

# 1. PyTorch 버전 출력
# 버전 뒤에 +cuXXX 와 같은 텍스트가 붙어있으면 CUDA 지원 버전일 확률이 높습니다.
print(f"PyTorch 버전: {torch.__version__}")

# 2. CUDA 사용 가능 여부 확인 (가장 중요한 확인 방법)
# 이 값이 True이면 GPU를 사용할 수 있습니다.
is_available = torch.cuda.is_available()
print(f"CUDA 사용 가능 여부: {is_available}")

# 3. 사용 가능하다면, PyTorch가 빌드된 CUDA 버전 확인
if is_available:
    print(f"PyTorch용 CUDA 버전: {torch.version.cuda}")
    print(f"연결된 GPU 이름: {torch.cuda.get_device_name(0)}")