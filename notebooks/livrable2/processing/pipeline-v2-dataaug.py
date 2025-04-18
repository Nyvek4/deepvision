"""
Nouvelle Pipeline V2 - Génération de bruits paramétrables
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

# Configuration des paramètres de bruit
NOISE_CONFIG = {
    'gaussian': {
        'mean': 0.08,      # Variance moyenne typique
        'std': 0.03,       # Écart-type pour la variance
        'clip': (0, 0.3)   # Plage de valeurs valides
    },
    'salt_pepper': {
        'mean': 0.04,      # Densité moyenne typique
        'std': 0.015,
        'clip': (0, 0.2)
    },
    'poisson': {},         # Bruit inhérent au capteur photo
    'speckle': {
        'mean': 0.12,
        'std': 0.05,
        'clip': (0, 0.4)
    }
}

SAMPLES_PER_NOISE_TYPE = 5  # Nombre de variations par type de bruit

def add_noise(image, noise_type, level=None):
    """Applique un type de bruit spécifique à l'image"""
    if noise_type == 'gaussian':
        variance = level * 255**2
        sigma = np.sqrt(variance)
        gauss = np.random.normal(0, sigma, image.shape)
        return np.clip(image + gauss, 0, 255).astype(np.uint8)
    
    elif noise_type == 'salt_pepper':
        prob = level
        output = np.copy(image)
        thres = 1 - prob
        random_matrix = np.random.rand(*image.shape)
        output[random_matrix < prob] = 0
        output[random_matrix > thres] = 255
        return output.astype(np.uint8)
    
    elif noise_type == 'poisson':
        vals = len(np.unique(image))
        vals = 2**np.ceil(np.log2(vals))
        return np.random.poisson(image * vals) / float(vals)
    
    elif noise_type == 'speckle':
        return np.clip(image + image * np.random.randn(*image.shape) * level, 0, 255).astype(np.uint8)

def generate_noise_level(config):
    """Génère un niveau de bruit suivant une distribution normale"""
    level = np.random.normal(config['mean'], config['std'])
    return np.clip(level, *config['clip'])

def process_images(input_dir, output_dir):
    """Pipeline principale de traitement"""
    
    # Création des dossiers
    clean_dir = os.path.join(output_dir, 'clean')
    noisy_dir = os.path.join(output_dir, 'noisy')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    
    metadata = []
    
    # Parcours des images originales
    for img_name in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        
        # Sauvegarde de l'image originale
        clean_path = os.path.join(clean_dir, img_name)
        cv2.imwrite(clean_path, image)
        
        # Génération des variations bruitées
        for noise_type, config in NOISE_CONFIG.items():
            # Cas particulier pour le bruit de Poisson
            if noise_type == 'poisson':
                noisy_image = add_noise(image, noise_type)
                noisy_name = f"{os.path.splitext(img_name)[0]}_{noise_type}.png"
                noisy_path = os.path.join(noisy_dir, noisy_name)
                cv2.imwrite(noisy_path, noisy_image)
                metadata.append([img_name, noisy_name, noise_type, None])
                continue
                
            # Génération de plusieurs niveaux de bruit
            for _ in range(SAMPLES_PER_NOISE_TYPE):
                level = generate_noise_level(config)
                noisy_image = add_noise(image, noise_type, level)
                
                # Formatage du nom de fichier
                noisy_name = f"{os.path.splitext(img_name)[0]}_{noise_type}_{level:.4f}.png"
                noisy_path = os.path.join(noisy_dir, noisy_name)
                
                cv2.imwrite(noisy_path, noisy_image)
                metadata.append([img_name, noisy_name, noise_type, level])
    
    # Sauvegarde des métadonnées
    df = pd.DataFrame(metadata, columns=['original', 'noisy', 'noise_type', 'level'])
    df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)

# Exemple d'utilisation
if __name__ == "__main__":
    process_images(
        input_dir="/mnt/c/Users/NyveK/Downloads/Datasets/Datasets/livrable2/Dataset",
        output_dir="/home/kevin/datasets/livrable2/processed_dataaug"
    )"""
Nouvelle Pipeline V2 - Génération de bruits paramétrables
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

# Configuration des paramètres de bruit
NOISE_CONFIG = {
    'gaussian': {
        'mean': 0.08,      # Variance moyenne typique
        'std': 0.03,       # Écart-type pour la variance
        'clip': (0, 0.3)   # Plage de valeurs valides
    },
    'salt_pepper': {
        'mean': 0.04,      # Densité moyenne typique
        'std': 0.015,
        'clip': (0, 0.2)
    },
    'poisson': {},         # Bruit inhérent au capteur photo
    'speckle': {
        'mean': 0.12,
        'std': 0.05,
        'clip': (0, 0.4)
    }
}

SAMPLES_PER_NOISE_TYPE = 5  # Nombre de variations par type de bruit

def add_noise(image, noise_type, level=None):
    """Applique un type de bruit spécifique à l'image"""
    if noise_type == 'gaussian':
        variance = level * 255**2
        sigma = np.sqrt(variance)
        gauss = np.random.normal(0, sigma, image.shape)
        return np.clip(image + gauss, 0, 255).astype(np.uint8)
    
    elif noise_type == 'salt_pepper':
        prob = level
        output = np.copy(image)
        thres = 1 - prob
        random_matrix = np.random.rand(*image.shape)
        output[random_matrix < prob] = 0
        output[random_matrix > thres] = 255
        return output.astype(np.uint8)
    
    elif noise_type == 'poisson':
        vals = len(np.unique(image))
        vals = 2**np.ceil(np.log2(vals))
        return np.random.poisson(image * vals) / float(vals)
    
    elif noise_type == 'speckle':
        return np.clip(image + image * np.random.randn(*image.shape) * level, 0, 255).astype(np.uint8)

def generate_noise_level(config):
    """Génère un niveau de bruit suivant une distribution normale"""
    level = np.random.normal(config['mean'], config['std'])
    return np.clip(level, *config['clip'])

def process_images(input_dir, output_dir):
    """Pipeline principale de traitement"""
    
    # Création des dossiers
    clean_dir = os.path.join(output_dir, 'clean')
    noisy_dir = os.path.join(output_dir, 'noisy')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    
    metadata = []
    
    # Parcours des images originales
    for img_name in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        
        # Sauvegarde de l'image originale
        clean_path = os.path.join(clean_dir, img_name)
        cv2.imwrite(clean_path, image)
        
        # Génération des variations bruitées
        for noise_type, config in NOISE_CONFIG.items():
            # Cas particulier pour le bruit de Poisson
            if noise_type == 'poisson':
                noisy_image = add_noise(image, noise_type)
                noisy_name = f"{os.path.splitext(img_name)[0]}_{noise_type}.png"
                noisy_path = os.path.join(noisy_dir, noisy_name)
                cv2.imwrite(noisy_path, noisy_image)
                metadata.append([img_name, noisy_name, noise_type, None])
                continue
                
            # Génération de plusieurs niveaux de bruit
            for _ in range(SAMPLES_PER_NOISE_TYPE):
                level = generate_noise_level(config)
                noisy_image = add_noise(image, noise_type, level)
                
                # Formatage du nom de fichier
                noisy_name = f"{os.path.splitext(img_name)[0]}_{noise_type}_{level:.4f}.png"
                noisy_path = os.path.join(noisy_dir, noisy_name)
                
                cv2.imwrite(noisy_path, noisy_image)
                metadata.append([img_name, noisy_name, noise_type, level])
    
    # Sauvegarde des métadonnées
    df = pd.DataFrame(metadata, columns=['original', 'noisy', 'noise_type', 'level'])
    df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)

# Exemple d'utilisation
if __name__ == "__main__":
    process_images(
        input_dir="/mnt/c/Users/NyveK/Downloads/Dataset Livrable 2/Dataset",
        output_dir="/home/kevin/datasets/livrable2/processed_dataaug"
    )