import torch
import psutil
import time
import os

try:
    while True:
        # Clear terminal
        os.system("cls" if os.name == "nt" else "clear")

        print("=" * 60)
        print("LIVE SYSTEM & GPU MEMORY MONITOR")
        print("=" * 60)

        # ---------------- SYSTEM RAM ----------------
        ram = psutil.virtual_memory()

        print("\nSYSTEM RAM")
        print("-" * 60)
        print(f"Total RAM     : {ram.total / (1024**3):.2f} GB")
        print(f"Used RAM      : {ram.used / (1024**3):.2f} GB")
        print(f"Available RAM : {ram.available / (1024**3):.2f} GB")
        print(f"RAM Usage     : {ram.percent:.1f}%")

        # ---------------- GPU ----------------
        print("\nGPU MEMORY")
        print("-" * 60)

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)

                total = props.total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)

                free = total - reserved
                usage = (reserved / total) * 100

                print(f"\nGPU {i}: {props.name}")
                print(f"Total VRAM     : {total:.2f} GB")
                print(f"Used/Reserved  : {reserved:.2f} GB")
                print(f"Allocated      : {allocated:.2f} GB")
                print(f"Free           : {free:.2f} GB")
                print(f"VRAM Usage     : {usage:.1f}%")

        else:
            print("CUDA/GPU not available")

        print("\n" + "=" * 60)
        print("Refreshing every 1 second... Press Ctrl+C to stop.")
        print("=" * 60)

        time.sleep(1)

except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")