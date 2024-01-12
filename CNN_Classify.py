import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# split the flower image files to train/test/val as a ratio of 0.8:0.1:0.1

# DataSet_Path = './flowers/'
# splitfolders.ratio(DataSet_Path, output='output', seed=99, ratio=(0.8, 0.1, 0.1))

# Set the parameters
batch_size = 64
img_width = 64
img_height = 64
epochs = 50

# Set the augmentation parameter
zoom_range = 0.2
shear_range = 0.2
height_shift_range = 0.3
width_shift_range = 0.2
rotation_range = 0.4

# Making the Image Data Generator for the dataset
train_path = './output/train'
test_path = './output/test'
val_path = './output/val'

# initialize ImageDataGenerator with augmentation
train_data_gen = ImageDataGenerator(rescale=1. / 255.,
                                    rotation_range=rotation_range,
                                    width_shift_range=width_shift_range,
                                    height_shift_range=height_shift_range,
                                    shear_range=shear_range,
                                    zoom_range=zoom_range,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

train_img_generator = train_data_gen.flow_from_directory(directory=train_path,
                                                         batch_size=batch_size,
                                                         class_mode="categorical",
                                                         target_size=(img_width, img_height))

test_data_gen = ImageDataGenerator(rescale=1. / 255.)

test_img_generator = test_data_gen.flow_from_directory(directory=test_path,
                                                       batch_size=batch_size,
                                                       class_mode="categorical",
                                                       target_size=(img_width, img_height)
                                                       )

val_data_gen = ImageDataGenerator(rescale=1. / 255.)

val_img_generator = val_data_gen.flow_from_directory(directory=val_path,
                                                     batch_size=batch_size,
                                                     class_mode="categorical",
                                                     target_size=(img_width, img_height)
                                                     )

# define a simple CNN architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(512, activation='relu', name='dense1'),
    tf.keras.layers.Dense(128, activation='relu', name='dense2'),
    tf.keras.layers.Dense(14, activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_img_generator,
                    epochs=epochs,
                    validation_data=val_img_generator,
                    verbose=2
                    )

# save the cnn model
model.save('./cnn_model')
