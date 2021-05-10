import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

main_dir = 'dogs-vs-cats'
train_dir = os.path.join(main_dir, 'train')
val_dir = os.path.join(main_dir, 'test1')

print('Training Data:')
print(len(os.listdir(os.path.join(train_dir, 'train'))))
print('Validation Data:')
print(len(os.listdir(os.path.join(val_dir, 'test1'))))

training_datagen = ImageDataGenerator(rescale=1/255,
                                      height_shift_range=0.2,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      shear_range=0.2,
                                      horizontal_flip=0.2,
                                      zoom_range=0.2
                                      )

validation_datagen = ImageDataGenerator(rescale=1/255)


training_generator = training_datagen.flow_from_directory(
                        train_dir,
                        target_size=(150,150),
                        batch_size=50,
                        class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
                        val_dir,
                        target_size=(150,150),
                        batch_size=50,
                        class_mode='binary'
)


# model subclassing
class CNNblock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3):
        super(CNNblock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


model = tf.keras.Sequential(
    [
        CNNblock(32),
        CNNblock(64),
        CNNblock(128),
        CNNblock(128),
        layers.Flatten(),
        layers.Dense(1)
    ]
)
#model.summary()

model.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

model.fit(training_generator,
          epochs = 10,
          validation_data=validation_generator)
