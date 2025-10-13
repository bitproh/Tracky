import torch
import numpy as np
import time

def demonstrate_gpu_acceleration():
    """
    A function to demonstrate the performance difference between CPU and GPU
    for a large numerical computation.
    """
    # 1. --- DEVICE SETUP ---
    # This is the most important step. We check if a CUDA-enabled GPU is available.
    # If yes, we select it; otherwise, we fall back to the CPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. GPU acceleration will not be demonstrated.")

    # 2. --- DATA PREPARATION ---
    # We'll create a large matrix to ensure the computation is intensive enough
    # to see a significant difference in performance.
    matrix_size = 10000
    print(f"\nCreating a {matrix_size}x{matrix_size} matrix...")
    
    # Create the data on the CPU first using NumPy
    numpy_array = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    # 3. --- CPU COMPUTATION ---
    print("Performing computation on CPU with NumPy...")
    start_time_cpu = time.time()
    
    # Perform a series of element-wise operations
    result_cpu = np.sin(numpy_array) * np.cos(numpy_array) + np.sqrt(numpy_array)
    
    end_time_cpu = time.time()
    cpu_duration = end_time_cpu - start_time_cpu
    print(f"CPU time: {cpu_duration:.4f} seconds")

    # 4. --- GPU COMPUTATION ---
    if device == 'cuda':
        print("\nPerforming the same computation on GPU with PyTorch...")
        
        # Move the data from CPU memory to the GPU's memory
        torch_tensor = torch.from_numpy(numpy_array).to(device)

        # It's good practice to run a "warm-up" operation on the GPU
        # to ensure all CUDA kernels are initialized before we start timing.
        _ = torch.sin(torch_tensor) * torch.cos(torch_tensor)
        torch.cuda.synchronize()

        start_time_gpu = time.time()
        
        # Perform the exact same operations on the GPU tensor
        result_gpu = torch.sin(torch_tensor) * torch.cos(torch_tensor) + torch.sqrt(torch_tensor)
        
        # IMPORTANT: GPU operations are asynchronous. We must explicitly wait for
        # the computation to finish on the GPU before stopping the timer.
        torch.cuda.synchronize()
        
        end_time_gpu = time.time()
        gpu_duration = end_time_gpu - start_time_gpu
        print(f"GPU time: {gpu_duration:.4f} seconds")

        # 5. --- RESULTS ---
        speedup = cpu_duration / gpu_duration
        print(f"\nGPU was approximately {speedup:.2f} times faster than the CPU for this task.")

if __name__ == "__main__":
    demonstrate_gpu_acceleration()
