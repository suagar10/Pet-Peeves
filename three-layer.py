#Importing Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

#Initializing the CNN
classifier=Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation = 'relu'))
classifier.add(Dense(units=1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()

#Fitting to Images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

validation_generator=test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=2000)

classifier.save('three-layer.h5')
classifier=load_model('three-layer.h5')

from tensorflow.keras.preprocessing import image
import numpy as np

testing=image.load_img('testImage.jpg', target_size = (128, 128))
testing=image.img_to_array(testing)
testing=np.expand_dims(testing, axis=0)
result=classifier.predict(testing)
if result[0][0]>=0.5:
        print('Its a Dog')
else:
        print('Its a Cat')