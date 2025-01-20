import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


class HieroglyphClassifier:
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        base_model = VGG19(weights='imagenet',
                           include_top=False, input_shape=self.input_shape)
        for layer in base_model.layers:
            layer.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs=base_model.input, outputs=predictions)

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def prepare_data(self, train_dir, valid_dir, test_dir, batch_size=32):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        valid_test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        valid_generator = valid_test_datagen.flow_from_directory(
            valid_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        test_generator = valid_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        return train_generator, valid_generator, test_generator

    def train(self, train_generator, valid_generator, epochs=10, callbacks=None):
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5', save_best_only=True),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True)
            ]
        history = self.model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        return history

    def evaluate(self, test_generator):
        return self.model.evaluate(test_generator)

    def predict_image(self, image_path):
        from tensorflow.keras.preprocessing import image
        img = image.load_img(image_path, target_size=self.input_shape[:2])
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = self.model.predict(img_array)
        return {
            'class_index': np.argmax(predictions[0]),
            'confidence': float(np.max(predictions[0]))
        }
