"""numeric_parameters.py"""


class Numerics:
    """
    Class to hold numerical parameters
    """
    def __init__(self):
        self.r_tol = 1.e-8
        self.a_tol = 1.e-8
        self.max_time_steps = 500
        self.max_step_rejections = 10000
        self.max_snes_failures = 20000 