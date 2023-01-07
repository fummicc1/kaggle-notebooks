import numpy as np
import torch
import os

def set_seed(seed):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state
