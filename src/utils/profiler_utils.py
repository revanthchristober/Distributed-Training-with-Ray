import cProfile
import pstats
import io
import time
import torch
import logging
import tracemalloc
from functools import wraps
from memory_profiler import memory_usage
from torch.profiler import profile, record_function, ProfilerActivity

# Logger setup
logger = logging.getLogger('profiler_utils')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def profile_cpu(func):
    """
    Decorator for profiling CPU performance of a function using cProfile.
    
    Parameters:
    - func (function): The function to be profiled.

    Returns:
    - wrapper (function): The wrapped function with profiling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        # Save profile stats to a string
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()

        # Log the profiler stats
        logger.info(f"CPU Profile Stats for {func.__name__}:\n{s.getvalue()}")
        return result
    return wrapper


def profile_memory(func):
    """
    Decorator for profiling memory usage of a function using memory_profiler.
    
    Parameters:
    - func (function): The function to be profiled.

    Returns:
    - wrapper (function): The wrapped function with memory profiling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        mem_usage = memory_usage((func, args, kwargs), max_iterations=1, retval=True)
        result, peak_memory = mem_usage[0], max(mem_usage[1:])
        
        # Log the peak memory usage
        logger.info(f"Peak memory usage for {func.__name__}: {peak_memory} MiB")
        return result
    return wrapper


def start_memory_profiling():
    """
    Start tracking memory usage using tracemalloc.
    
    Returns:
    - None
    """
    tracemalloc.start()
    logger.info("Started memory profiling using tracemalloc.")


def stop_memory_profiling():
    """
    Stop tracking memory usage and display top memory-consuming operations.
    
    Returns:
    - None
    """
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    logger.info("[Top 10 Memory-consuming operations]")
    for stat in top_stats[:10]:
        logger.info(stat)

    tracemalloc.stop()


def profile_gpu(model, inputs, activities=None, output_file=None, warmup_steps=5, active_steps=10):
    """
    Profile the GPU performance using PyTorch's torch.profiler.
    
    Parameters:
    - model (torch.nn.Module): The model to be profiled.
    - inputs (torch.Tensor): Example input data to feed into the model for profiling.
    - activities (list): List of ProfilerActivity (CUDA, CPU) to profile.
    - output_file (str): Path to save the profiling results.
    - warmup_steps (int): Number of warmup steps.
    - active_steps (int): Number of steps to profile.
    
    Returns:
    - None
    """
    if activities is None:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    model.eval()  # Set the model to evaluation mode
    logger.info(f"Profiling GPU performance for {model.__class__.__name__}...")

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(warmup_steps + active_steps):
            with record_function("model_inference"):
                output = model(inputs)

    # Print profiler stats
    logger.info(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    if output_file:
        prof.export_chrome_trace(output_file)
        logger.info(f"Profiling output saved to {output_file}")


def profile_execution_time(func):
    """
    Decorator to profile the execution time of a function.
    
    Parameters:
    - func (function): The function to be profiled.

    Returns:
    - wrapper (function): The wrapped function with execution time profiling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        exec_time = end_time - start_time
        logger.info(f"Execution time for {func.__name__}: {exec_time:.4f} seconds")
        return result
    return wrapper


def detailed_memory_profile(func):
    """
    Decorator for detailed line-by-line memory profiling using memory_profiler.
    
    Parameters:
    - func (function): The function to be profiled.

    Returns:
    - wrapper (function): The wrapped function with line-by-line memory profiling.
    """
    from memory_profiler import profile as mem_profile

    @wraps(func)
    @mem_profile
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def profile_torch_autograd(model, inputs):
    """
    Profile autograd for CPU/GPU bottlenecks using PyTorch's autograd profiler.
    
    Parameters:
    - model (torch.nn.Module): The model to profile.
    - inputs (torch.Tensor): Example inputs for the model.

    Returns:
    - None
    """
    logger.info(f"Profiling autograd for {model.__class__.__name__}...")
    
    with torch.autograd.profiler.profile(record_shapes=True, use_cuda=True) as prof:
        output = model(inputs)

    logger.info(prof.key_averages().table(sort_by="cuda_time_total"))
    

def model_size(model: torch.nn.Module):
    """
    Compute the size of a model in megabytes.

    Parameters:
    - model (torch.nn.Module): The PyTorch model.

    Returns:
    - size (float): Size of the model in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size = (param_size + buffer_size) / (1024 ** 2)
    logger.info(f"Model {model.__class__.__name__} size: {size:.4f} MB")
    return size


# Example usage for profiling a PyTorch model
if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # Define a simple PyTorch model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(512, 10)

        def forward(self, x):
            return self.fc(x)

    # Example model and inputs
    model = SimpleModel()
    inputs = torch.randn(16, 512)

    # GPU profiling
    profile_gpu(model, inputs, output_file="gpu_profile.json")

    # Memory profiling
    @profile_memory
    def model_inference():
        return model(inputs)

    model_inference()

    # CPU profiling
    @profile_cpu
    def cpu_heavy_operation():
        for _ in range(1000000):
            pass

    cpu_heavy_operation()

    # Model size
    model_size(model)
