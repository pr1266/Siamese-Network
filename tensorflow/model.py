import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
class SiameseModel(Model):

    def __init__(self):

        super(SiameseModel, self).__init__()
        self.feature_extractor = Sequential([
            Conv2D(96, kernel_size=11, strides=4, activation='relu',),
            MaxPool2D(pool_size=3, strides=2),
            #**************
            Conv2D(256, kernel_size=5, strides=1, activation='relu',),
            MaxPool2D(pool_size=2, strides=2),
            #**************
            Conv2D(384, kernel_size=3, strides=1, activation='relu',),
        ])
        
        self.classifier = Sequential([
            Dense(1024, activation='relu',),
            Dense(256, activation='relu',),
            Dense(2, activation='relu',)
        ])

    def call(self, x):
        out = self.feature_extractor(x)
        out = self.classifier(out)
        return out

model = SiameseModel()
print(model)