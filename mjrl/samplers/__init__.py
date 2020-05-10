import numpy as np

def get_tajectories_per_cpu(num_traj, num_cpu):
    return int(np.ceil(num_traj/num_cpu))
