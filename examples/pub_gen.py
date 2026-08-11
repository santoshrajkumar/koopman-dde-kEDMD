"""
This script reproduces the figures in the paper.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kedmd_dde.Utils import pub_gen


pub_gen.m_vary()
pub_gen.p_vary()
pub_gen.rho_vary()
pub_gen.scalar_phase()
pub_gen.planar_phase()

pub_gen.gen_data()
