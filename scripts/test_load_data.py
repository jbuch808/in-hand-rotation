import numpy as np
import torch

# Data Mapping
# 0 -> obs
# 1 -> actions
# 2 -> sigmas
# 3 -> pointCloud BxNx6 (Last dimension: 1-3 xyz/4-6 point type)
#   - 100: imagined point
#   - 010: camera point
#   - 001: tactile point
# 4 -> batch_depth_image
# 5 -> imagined_pc
# 6 -> fsr_pc


def main():
    data_path = '/workspace/code/in-hand-rotation/demonstration-baoding/teacher_batch_0_0.pt'

    data = torch.load(data_path)
    pc_all = data[3]
    pc = pc_all[2000]
    mask_camera = pc[:, 4] == 1
    pc_camera = pc[mask_camera, 0:3]

if __name__ == '__main__':
    main()
