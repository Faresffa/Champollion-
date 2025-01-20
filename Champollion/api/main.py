from flask import Flask, request, jsonify
import torch
from PIL import Image
# Import de votre modèle
from model import HieroglyphClassifier

app = Flask(__name__)


train_path = 'dataset/train'
valid_path = 'dataset/valid'
test_path = 'dataset/test'

classifier = HieroglyphClassifier(num_classes=3, input_shape=(224, 224, 3))
classifier.compile_model()

train_gen, valid_gen, test_gen = classifier.prepare_data(
    train_dir=train_path,
    valid_dir=valid_path,
    test_dir=test_path
)
test_loss, test_accuracy = classifier.evaluate(test_gen)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

result = classifier.predict_image('path_to_image.jpg')
print(f"Predicted class: {result['class_index']
                          }, Confidence: {result['confidence']}")


# Charger votre modèle pré-entraîné
model = HieroglyphClassifier()
model.load_state_dict(torch.load('path/to/your/model.pth'))
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Charger et prétraiter l'image
    image = Image.open(request.files['image'])
    # Ajouter votre logique de prétraitement ici

    # Faire la prédiction
    with torch.no_grad():
        prediction = model(image)

    # Retourner le résultat
    return jsonify({
        'class': prediction.class_name,
        'confidence': float(prediction.confidence),
        'description': get_hieroglyph_description(prediction.class_name)
    })


if __name__ == '__main__':
    app.run(debug=True)
