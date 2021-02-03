
import os

def get_available_gpuid():
    available_gpuid = range(0, 8)
    for line in os.popen('nvidia-smi').readlines():
        if "python" not in line: continue
        # |    5     63919      C   python                                       184MiB |
        gpuid, pid, C, py, mem = line.strip().split()[1:-1]
        mem_val = mem[0:len(mem)-3]
        if int(mem_val) > 100:
            available_gpuid.remove(int(gpuid))
    return available_gpuid

