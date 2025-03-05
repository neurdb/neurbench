import random
import os
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
seed_everything(42)

sample_rng = np.random.default_rng(np.random.MT19937(seed=np.random.randint(0, 2**32)))
