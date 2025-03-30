import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.15
)

train_generator = train_datagen.flow_from_directory(
    'asl_data/asl_alphabet_train/asl_alphabet_train',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)


validation_generator = train_datagen.flow_from_directory(
    'asl_data/asl_alphabet_train/asl_alphabet_train',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Input(shape=(200,200,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.40),
    Dense(29, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator
)

model.save('asl_model.h5')

# test_loss, test_accuracy = model.evaluate(test_generator)  # or (X_test, y_test)
# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_accuracy)
