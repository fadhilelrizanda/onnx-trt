import pycuda.driver as cuda
import pycuda.autoinit

def list_gpus():
    num_gpus = cuda.Device.count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        gpu = cuda.Device(i)
        print(f"GPU {i}: {gpu.name()}")
        print(f"  Compute Capability: {gpu.compute_capability()}")
        print(f"  Total Memory: {gpu.total_memory() // (1024 ** 2)} MB")

list_gpus()