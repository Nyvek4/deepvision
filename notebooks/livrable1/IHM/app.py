import os
import datetime
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# Fonction de chargement et configuration du modèle
def load_and_configure_model(model_path=None, model_dir='models/5cv1', force_reload=False):
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le modèle spécifié n'existe pas: {model_path}")
        final_model_path = model_path
    else:
        if hasattr(load_and_configure_model, 'cached_model_path') and not force_reload:
            print(f"Utilisation du modèle en cache: {load_and_configure_model.cached_model_path}")
            final_model_path = load_and_configure_model.cached_model_path
        else:
            model_files = []
            for ext in ['.tflite', '.h5', '.keras']:
                model_files.extend([f for f in os.listdir(model_dir) if f.endswith(ext)])
            if not model_files:
                raise FileNotFoundError(f"Aucun modèle trouvé dans {model_dir}")
            timestamps = []
            for file in model_files:
                try:
                    timestamp_str = file.split('_')[-1].split('.')[0]
                    timestamp = datetime.datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S')
                    timestamps.append((file, timestamp))
                except (ValueError, IndexError):
                    print(f"Format de nom de fichier non reconnu pour {file}, ignoré.")
            timestamps.sort(key=lambda x: x[1], reverse=True)
            if not timestamps:
                raise ValueError("Aucun modèle avec un format de timestamp valide n'a été trouvé.")
            latest_model_file = timestamps[0][0]
            final_model_path = os.path.join(model_dir, latest_model_file)
            load_and_configure_model.cached_model_path = final_model_path
    is_tflite = final_model_path.endswith('.tflite')
    model_name = os.path.basename(final_model_path).lower()
    if 'resnet' in model_name:
        print("Modèle détecté: ResNet - Prétraitement ResNet")
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    elif 'efficient' in model_name:
        print("Modèle détecté: EfficientNet - Prétraitement EfficientNet")
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif 'xception' in model_name:
        print("Modèle détecté: Xception - Prétraitement Xception")
        preprocess_input = tf.keras.applications.xception.preprocess_input
    else:
        print("Type de modèle non identifié - Utilisation du prétraitement ResNet par défaut")
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    print(f"Modèle chargé: {final_model_path}")
    print(f"Format: {'TFLite' if is_tflite else 'Keras'}")
    return final_model_path, preprocess_input, is_tflite

# Fonction qui lit une image (données bytes) et renvoie la prédiction
def get_image_prediction_from_bytes(file_bytes, model, class_names, is_tflite, preprocess_input):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image invalide ou impossible à décoder.")
    if img.shape[0:2] != (224, 224):
        img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_img = preprocess_input(img.astype(np.float32))
    processed_img = np.expand_dims(processed_img, axis=0)
    if is_tflite:
        interpreter = tf.lite.Interpreter(model_path=model if isinstance(model, str) else None)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]['index'])[0]
    else:
        probs = model.predict(processed_img, verbose=0)[0]
    predicted_class = class_names[np.argmax(probs)]
    return predicted_class, probs

app = Flask(__name__)

# Route pour la page d'accueil qui sert l'IHM
@app.route('/')
def index():
    return render_template('index.html')

# Chargement du modèle une seule fois au démarrage
MODEL_DIR = '../models/5cv1'
MODEL_PATH, preprocess_input, is_tflite = load_and_configure_model(model_dir=MODEL_DIR)
if is_tflite:
    model = MODEL_PATH  # Pour TFLite, seul le chemin est nécessaire
else:
    model = tf.keras.models.load_model(MODEL_PATH)

# Définition des classes attendues
CLASS_NAMES = ['Painting', 'Photo', 'Schematics', 'Sketch', 'Text']

# Route API pour la prédiction en POST
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': "Aucune image fournie."}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': "Nom de fichier vide."}), 400
    try:
        file_bytes = file.read()
        predicted_class, probs = get_image_prediction_from_bytes(
            file_bytes, model, CLASS_NAMES, is_tflite, preprocess_input
        )
        prob_dict = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}
        return jsonify({
            'predicted_class': predicted_class,
            'probabilities': prob_dict
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
