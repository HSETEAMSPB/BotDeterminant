from config import *

import tensorflow as tf

class MakeLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self):
        super(MakeLR, self).__init__()
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.multiplier = peak_lr * (warmup_steps ** 0.5)
    
    @tf.function
    def __call__(self, step: float):
        if step < self.warmup_steps:
            lr = float(step) * (float(self.warmup_steps) ** (-1.5))
        else:
            lr = float(step) ** (-0.5)
        
        return tf.math.minimum(self.multiplier * lr, self.peak_lr)

def make_opt():
    lr = MakeLR()
    return tf.keras.optimizers.Adam(learning_rate=lr)
