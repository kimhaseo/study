# units.py
import numpy as np

MM_TO_M = 1e-3
M_TO_MM = 1e3

def m_to_mm_vec(v_m: np.ndarray) -> np.ndarray:
    return v_m * M_TO_MM

def mm_to_m_vec(v_mm: np.ndarray) -> np.ndarray:
    return v_mm * MM_TO_M
