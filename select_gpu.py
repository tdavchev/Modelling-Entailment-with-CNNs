import sys
from pynvml import *
try:
    nvmlInit()
except NVMLError as err:
    print "Failed to initialize NVML: " % err
    print "Exiting..."
    os._exit(1)
num_gpus = int(nvmlDeviceGetCount())
stats = []
for gpu_id in xrange(num_gpus):
    gpu_obj = nvmlDeviceGetHandleByIndex(gpu_id)
    used_mem = nvmlDeviceGetMemoryInfo(gpu_obj).used
    stats.append((gpu_id, used_mem))
stats = sorted(stats, key=lambda x: x[1])
sys.stdout.write("%s" % stats[0][0])
try:
    nvmlShutdown()
except NVMLError as err:
    print "Error shutting down NVML:" % err
    os._exit(1)