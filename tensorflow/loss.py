import tensorflow as tf
from tensorflow.keras.losses import Loss

class ContrastiveLoss(Loss):

  def __init__(self):
    super().__init__()

  def call(self, y_true, y_pred):
    pass