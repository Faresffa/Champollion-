import tensorflow as tf
from model import HieroglyphClassifier

# Exemple d'utilisation


def train_hieroglyph_classifier():
    # Initialiser le modèle
    num_classes = 20  # Ajuster selon votre nombre de classes de hiéroglyphes
    classifier = HieroglyphClassifier(num_classes)

    # Compiler le modèle
    classifier.compile_model()

    # Préparer les données
    train_generator, validation_generator = classifier.prepare_data(
        train_dir='path/to/train',
        validation_dir='path/to/validation'
    )

    # Callbacks pour l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5', save_best_only=True)
    ]

    # Entraînement initial
    classifier.train(train_generator, validation_generator,
                     epochs=10, callbacks=callbacks)

    # Fine-tuning
    classifier.fine_tune()
    classifier.train(train_generator, validation_generator,
                     epochs=5, callbacks=callbacks)

    return classifier
