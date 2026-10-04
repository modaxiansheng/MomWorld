import runpy

import cv2
import torch


cv2.setNumThreads(1)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

runpy.run_path("tools/test.py", run_name="__main__")
