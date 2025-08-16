import numpy as np
import random

def tsmix_data(x, scale_decomp):
    lam1 = 1 - random.randint(1, 100) / 1000
    # lam2 = 1 - random.randint(1, 50) / 100

    short_term, long_term = scale_decomp(x)

    short_term_x = lam1 * x + (1 - lam1) * short_term
    long_term_x = lam1 * x +  (1 - lam1) * long_term

    return short_term_x, long_term_x

