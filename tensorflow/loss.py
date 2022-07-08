import tensorflow as tf
from tensorflow.keras.losses import Loss

class ContrastiveLoss(Loss):

  def __init__(self, margin):
    super().__init__()
    self.margin = margin

  def call(self, y_true, y_pred, label):
    
    euclidean_distance = tf.math.reduce_euclidean_norm(tf.constant([y_true, y_pred]))
    loss_contrastive = tf.mean((1-label) * tf.pow(euclidean_distance, 2) +
                                    (label) * tf.pow(tf.clip_by_value(self.margin - euclidean_distance, min=0.0), 2))

    return loss_contrastive
    