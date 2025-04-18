# %%
# %%
# %% [markdown]
# ## 1. Chargement des Bibliothèques
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from IPython.display import display
from math import ceil

# Reproductibilité
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# %%
# %%
# %% [markdown]
# ## 2. Paramètres et Configuration
IMG_SIZE = (256, 256) 
BATCH_SIZE = 16  # Augmenté pour mieux exploiter les nouvelles données
EPOCHS = 100
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
DATA_PATH = '/home/kevin/datasets/livrable2/processed_dataaug'
METADATA_PATH = os.path.join(DATA_PATH, 'metadata.csv')

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
    'poisson': {},         # Bruit inhérent au capteur
    'speckle': {
        'mean': 0.12,
        'std': 0.05,
        'clip': (0, 0.4)
    }
}

# %%
# %%
# %% [markdown]
# ## 3. Chargement des Données avec Métadonnées (Nouvelle Version)

def data_generator(image_paths, clean_paths, batch_size=32):
    while True:
        indices = np.random.permutation(len(image_paths))
        for i in range(0, len(indices), batch_size):
            batch_paths = indices[i:i+batch_size]
            noisy_batch = []
            clean_batch = []
            for idx in batch_paths:
                # Chargement, forçage couleur et resize
                noisy = cv2.imread(image_paths[idx])
                noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB) if noisy is not None else None
            
                clean = cv2.imread(clean_paths[idx])
                clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB) if clean is not None else None
                
                if noisy is None:
                    noisy = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
                else:
                    noisy = cv2.resize(noisy, IMG_SIZE)
                    noisy = noisy / 255.0
                if clean is None:
                    clean = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
                else:
                    clean = cv2.resize(clean, IMG_SIZE)
                    clean = clean / 255.0
                # Augmentation aléatoire
                if np.random.rand() > 0.5:
                    noisy = cv2.flip(noisy, 1)
                    clean = cv2.flip(clean, 1)
                if np.random.rand() > 0.5:
                    angle = np.random.randint(-15, 15)
                    M = cv2.getRotationMatrix2D((IMG_SIZE[0]//2, IMG_SIZE[1]//2), angle, 1)
                    noisy = cv2.warpAffine(noisy, M, IMG_SIZE)
                    clean = cv2.warpAffine(clean, M, IMG_SIZE)
                noisy_batch.append(noisy)
                clean_batch.append(clean)
            yield np.stack(noisy_batch), np.stack(clean_batch)

def prepare_generators(data_path, metadata_path, test_size=0.2, val_size=0.2, batch_size=32):
    # Charger les métadonnées
    metadata = pd.read_csv(metadata_path)
    
    # Créer les listes de chemins complets
    image_paths = [os.path.join(data_path, 'noisy', f) for f in metadata['noisy']]
    clean_paths = [os.path.join(data_path, 'clean', f) for f in metadata['original']]
    
    # Split initial train/test
    train_paths, test_paths, train_clean, test_clean = train_test_split(
        image_paths, clean_paths, 
        test_size=test_size, 
        random_state=SEED
    )
    
    # Split train/val
    train_paths, val_paths, train_clean, val_clean = train_test_split(
        train_paths, train_clean,
        test_size=val_size,
        random_state=SEED
    )
    
    return {
        'train': data_generator(train_paths, train_clean, BATCH_SIZE),
        'val': data_generator(val_paths, val_clean, BATCH_SIZE),
        'test': (
            np.array([
                cv2.resize(cv2.imread(p, cv2.IMREAD_COLOR), IMG_SIZE)/255.0
                if cv2.imread(p, cv2.IMREAD_COLOR) is not None else np.zeros((*IMG_SIZE, 3))
                for p in test_paths
            ]),
            np.array([
                cv2.resize(cv2.imread(p, cv2.IMREAD_COLOR), IMG_SIZE)/255.0
                if cv2.imread(p, cv2.IMREAD_COLOR) is not None else np.zeros((*IMG_SIZE, 3))
                for p in test_clean
            ])
        ),
        'steps': {
            'train': len(train_paths) // BATCH_SIZE,
            'val': len(val_paths) // BATCH_SIZE
        }
    }

# Chargement avec métadonnées
try:
    data = prepare_generators(DATA_PATH, METADATA_PATH, test_size=TEST_SPLIT, val_size=VAL_SPLIT, batch_size=BATCH_SIZE)
except Exception as e:
    print(f"Erreur: {e}")
    raise

# Accès aux données
train_gen = data['train']
val_gen = data['val']
(X_noisy, X_clean) = data['test']
steps = data['steps']


# %%
# %%
# %% [markdown]
# ## 4. Data Augmentation en Temps Réel

# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal_and_vertical"),
#     layers.RandomRotation(0.2),
#     layers.RandomZoom(0.1),
#     layers.RandomContrast(0.1)
# ])

# def preprocess_data(image, label):
#     # Application de l'augmentation seulement sur l'image d'entrée
#     augmented_image = data_augmentation(image)
#     return augmented_image, label

# # Création des datasets avec augmentation
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_dataset = train_dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
# train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
# val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# %%
# %%
# %% [markdown]
# ## 5. Architecture du Modèle (Améliorée)

class SpatialAttention(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.dense1 = layers.Dense(1, activation='sigmoid')

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, 7, padding='same', activation='sigmoid')
        super().build(input_shape)

    def call(self, x):
        # Attention spatiale
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        concat = layers.Concatenate()([max_pool, avg_pool])
        attention = self.conv(concat)
        return x * attention

def residual_block(x, filters, use_attention=False):
    # Project shortcut if input channels do not match output filters
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if use_attention:
        x = SpatialAttention()(x)
    
    x = layers.Add()([shortcut, x])
    return layers.Activation('relu')(x)

def multi_scale_unet(input_shape=(256,256,3)):
    inputs = layers.Input(input_shape)
    
    # Encoder multi-échelle
    e1 = residual_block(inputs, 64)
    p1 = layers.MaxPooling2D()(e1)
    
    e2 = residual_block(p1, 128, use_attention=True)
    p2 = layers.MaxPooling2D()(e2)
    
    e3 = residual_block(p2, 256, use_attention=True)
    p3 = layers.MaxPooling2D()(e3)
    
    # Bridge
    bridge = residual_block(p3, 512, use_attention=True)
    bridge = layers.Dropout(0.4)(bridge)
    
    # Decoder avec connexions denses
    d3 = layers.UpSampling2D()(bridge)
    d3 = layers.Concatenate()([d3, e3])
    d3 = residual_block(d3, 256)
    
    d2 = layers.UpSampling2D()(d3)
    d2 = layers.Concatenate()([d2, e2])
    d2 = residual_block(d2, 128)
    
    d1 = layers.UpSampling2D()(d2)
    d1 = layers.Concatenate()([d1, e1])
    d1 = residual_block(d1, 64)
    
    outputs = layers.Conv2D(3, 1, activation='sigmoid')(d1)
    
    return Model(inputs, outputs)

# %%
# 1. Définition des métriques de base
def PSNR(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def SSIM(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

# 2. Métrique hybride personnalisée
class HybridPSNRSSIM(tf.keras.metrics.Metric):
    def __init__(self, name='hybrid_psnr_ssim', **kwargs):
        super().__init__(name=name, **kwargs)
        self.psnr = tf.keras.metrics.Mean(name='psnr')
        self.ssim = tf.keras.metrics.Mean(name='ssim')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        self.psnr.update_state(tf.image.psnr(y_true, y_pred, 1.0))
        self.ssim.update_state(tf.image.ssim(y_true, y_pred, 1.0))
        
    def result(self):
        return (self.psnr.result() + self.ssim.result())/2
    
    def reset_states(self):
        self.psnr.reset_states()
        self.ssim.reset_states()

# 3. Perte hybride avancée avec intégration VGG
# Initialisation du modèle VGG16 (à placer avant la fonction de perte)
vgg = tf.keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3)
)
vgg.trainable = False

def hybrid_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Calcul des différentes composantes de la perte
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Pré-traitement pour VGG16 (mise à l'échelle [0,255])
    y_true_scaled = y_true * 255.0
    y_pred_scaled = y_pred * 255.0

    # Extraction des caractéristiques
    y_true_features = vgg(tf.keras.applications.vgg16.preprocess_input(y_true_scaled))
    y_pred_features = vgg(tf.keras.applications.vgg16.preprocess_input(y_pred_scaled))

    # Perte perceptuelle
    perceptual_loss = tf.reduce_mean(tf.square(y_true_features - y_pred_features))

    # Combinaison pondérée
    return 0.6 * ssim_loss + 0.3 * mse_loss + 0.1 * perceptual_loss

# %%
# %%
# %% [markdown]
# ## 6. Stratégie d'Entraînement Avancée

class NoiseAdaptiveLoss(tf.keras.losses.Loss):
    def __init__(self, noise_info, **kwargs):
        super().__init__(**kwargs)
        self.noise_levels = np.array([info[1] or 0 for info in noise_info])
        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.loss_fn(y_true, y_pred)
        # Pondération en fonction du niveau de bruit
        weighted_loss = loss * (1 + tf.cast(self.noise_levels, tf.float32))
        return tf.reduce_mean(weighted_loss)
    
# Callback personnalisé pour le monitoring des métriques par type de bruit
class NoiseTypeMonitor(callbacks.Callback):
    def __init__(self, X_val, y_val, noise_info):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.noise_info = noise_info

    def on_epoch_end(self, epoch, logs=None):
        # Évaluation par type de bruit
        for noise_type in NOISE_CONFIG.keys():
            idx = [i for i, info in enumerate(self.noise_info) if info[0] == noise_type]
            if len(idx) > 0:
                X_sub = self.X_val[idx]
                y_sub = self.y_val[idx]
                results = self.model.evaluate(X_sub, y_sub, verbose=0)
                logs[f'val_{noise_type}_loss'] = results[0]
                logs[f'val_{noise_type}_psnr'] = results[1]
                logs[f'val_{noise_type}_ssim'] = results[2]

# Charger les métadonnées
metadata = pd.read_csv(METADATA_PATH)

# Récupérer les chemins de validation
val_indices = list(range(steps['train'] * BATCH_SIZE, (steps['train'] + steps['val']) * BATCH_SIZE))
val_metadata = metadata.iloc[val_indices]

# Générer X_val, y_val et noise_info
X_val = []
y_val = []
noise_info = []
for _, row in val_metadata.iterrows():
    noisy_path = os.path.join(DATA_PATH, 'noisy', row['noisy'])
    clean_path = os.path.join(DATA_PATH, 'clean', row['original'])
    noisy_img = cv2.imread(noisy_path, cv2.IMREAD_COLOR)
    noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB) if noisy_img is not None else None  
    clean_img = cv2.imread(clean_path, cv2.IMREAD_COLOR)
    clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB) if clean_img is not None else None
    if noisy_img is not None and clean_img is not None:
        noisy_img = cv2.resize(noisy_img, IMG_SIZE) / 255.0
        clean_img = cv2.resize(clean_img, IMG_SIZE) / 255.0
        X_val.append(noisy_img)
        y_val.append(clean_img)
        # Suppose que le type de bruit et le niveau sont dans les colonnes 'noise_type' et 'noise_level'
        noise_info.append((row.get('noise_type', 'unknown'), row.get('noise_level', 0)))
X_val = np.array(X_val)
y_val = np.array(y_val)

# Optimiseur avec gradient clipping
# Optimiseur avec gradient clipping
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-3,
    weight_decay=1e-4,
    global_clipnorm=1.0
)

# Compilation
model = multi_scale_unet(input_shape=(256, 256, 3))
model.compile(
    optimizer=optimizer,
    loss=hybrid_loss,  # Utilisation de la perte hybride
    metrics=[
        PSNR,          # Métrique PSNR standard
        SSIM,          # Métrique SSIM standard
        HybridPSNRSSIM() # Métrique hybride personnalisée
    ]
)
# Définition des callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
checkpoint = callbacks.ModelCheckpoint(
    "models/best_model-v3-dataaug.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
tensorboard = callbacks.TensorBoard(
    log_dir="logs/fit/" + pd.Timestamp.now().strftime("%Y%m%d-%H%M%S"),
    histogram_freq=1
)


# %%

# Entraînement du modèle
history = model.fit(
    train_gen,
    steps_per_epoch=steps['train'],
    validation_data=val_gen,
    validation_steps=steps['val'],
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, tensorboard, NoiseTypeMonitor(X_val, y_val, noise_info)],
    verbose=1
)

# %%
import glob

### LOAD THE MODEL IF NEEDED ####

model = tf.keras.models.load_model(
    "models/best_model-v3-dataaug.h5",
    custom_objects={
        'HybridPSNRSSIM': HybridPSNRSSIM,
        'NoiseAdaptiveLoss': NoiseAdaptiveLoss,
        'PSNR': PSNR,
        'SSIM': SSIM,
        'hybrid_loss': hybrid_loss,
        'SpatialAttention': SpatialAttention,  # Ajout de la couche custom
        'residual_block': residual_block,      # (optionnel si utilisé dans le modèle)
        'multi_scale_unet': multi_scale_unet   # (optionnel si utilisé dans le modèle)
    }
)


# %%
# %%
# %% [markdown]
# ## 8. Évaluation Multi-Critères

import collections

(X_noisy, X_clean) = data['test']

def evaluate_by_noise_type(model, X_test, y_test, noise_info):
    """
    Évalue le modèle pour chaque type de bruit présent dans noise_info.
    """
    results = collections.OrderedDict()
    noise_types = list(NOISE_CONFIG.keys())
    # Regrouper les indices par type de bruit
    noise_indices = {nt: [] for nt in noise_types}
    for i, info in enumerate(noise_info):
        if info[0] in noise_types:
            noise_indices[info[0]].append(i)
    # Évaluer pour chaque type de bruit
    for noise_type, idxs in noise_indices.items():
        if not idxs:
            continue
        X_sub = X_test[idxs]
        y_sub = y_test[idxs]
        test_results = model.evaluate(X_sub, y_sub, verbose=0)
        results[noise_type] = {
            'loss': test_results[0],
            'psnr': test_results[1],
            'ssim': test_results[2]
        }
    return results

# Correction : utiliser X_noisy et X_clean pour l'évaluation
detailed_results = evaluate_by_noise_type(model, X_noisy, X_clean, noise_info)

# Affichage des résultats
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.bar(list(detailed_results.keys()), [v['psnr'] for v in detailed_results.values()])
plt.title('PSNR par Type de Bruit')
plt.ylabel('dB')

plt.subplot(1, 2, 2)
plt.bar(list(detailed_results.keys()), [v['ssim'] for v in detailed_results.values()])
plt.title('SSIM par Type de Bruit')
plt.ylabel('Score')
plt.show()

# %%
def plot_noise_comparison(noise_type='gaussian', n_samples=3):
    # Sélection d'images avec le type de bruit spécifié
    idx = [i for i, info in enumerate(noise_info) if info[0] == noise_type]
    if len(idx) == 0:
        print(f"Aucune image trouvée pour le bruit '{noise_type}'")
        return
    n_samples = min(n_samples, len(idx))
    samples = np.random.choice(idx, size=n_samples, replace=False)
    
    plt.figure(figsize=(15, 4 * n_samples))
    for i, sample_idx in enumerate(samples):
        # Récupération des données
        noisy = X_noisy[sample_idx].astype(np.float32)
        clean = X_clean[sample_idx].astype(np.float32)
        

        pred = model.predict(noisy[np.newaxis, ...])[0].astype(np.float32)
        
        # Calcul des métriques (forcer float32)
        psnr_val = tf.image.psnr(clean, pred, max_val=1.0).numpy()
        ssim_val = tf.image.ssim(clean, pred, max_val=1.0).numpy()
        
        # Affichage
        plt.subplot(n_samples, 3, i*3+1)
        plt.imshow(np.clip(noisy, 0, 1))
        plt.title(f"Bruité ({noise_type})")
        plt.axis('off')
        
        plt.subplot(n_samples, 3, i*3+2)
        plt.imshow(np.clip(pred, 0, 1))
        plt.title(f"Débruité\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.3f}")
        plt.axis('off')
        
        plt.subplot(n_samples, 3, i*3+3)
        plt.imshow(np.clip(clean, 0, 1))
        plt.title("Original")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Génération des visualisations pour chaque type de bruit
for noise_type in NOISE_CONFIG.keys():
    plot_noise_comparison(noise_type=noise_type)

# %%
import glob

#### LOAD THE MODEL

model = tf.keras.models.load_model(
    "models/best_model-v3-dataaug.h5",
    custom_objects={
        'HybridPSNRSSIM': HybridPSNRSSIM,
        'NoiseAdaptiveLoss': NoiseAdaptiveLoss,
        'PSNR': PSNR,
        'SSIM': SSIM,
        'hybrid_loss': hybrid_loss,
        'SpatialAttention': SpatialAttention,  # Ajout de la couche custom
        'residual_block': residual_block,      # (optionnel si utilisé dans le modèle)
        'multi_scale_unet': multi_scale_unet   # (optionnel si utilisé dans le modèle)
    }
)

def add_noise(img, noise_type, noise_level):
    img = img.astype(np.float32)
    if noise_type == 'gaussian':
        mean = 0
        std = noise_level
        noise = np.random.normal(mean, std, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 1)
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5
        amount = noise_level
        noisy = np.copy(img)
        # Salt
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape]
        noisy[tuple(coords)] = 1
        # Pepper
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape]
        noisy[tuple(coords)] = 0
        return noisy
    elif noise_type == 'poisson':
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
        return np.clip(noisy, 0, 1)
    elif noise_type == 'speckle':
        noise = np.random.randn(*img.shape) * noise_level
        noisy = img + img * noise
        return np.clip(noisy, 0, 1)
    else:
        return img

def test_model_on_deposit(model, deposit_dir="notebooks/livrable2/tests_deposit", n_variants=3):
    if not os.path.exists(deposit_dir):
        os.mkdir(deposit_dir)
        return

    image_paths = glob.glob(os.path.join(deposit_dir, "*"))
    if not image_paths:
        print("Aucune image trouvée dans", deposit_dir)
        return

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Impossible de lire {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE) / 255.0
        img = img.astype(np.float32)
        for noise_type, config in NOISE_CONFIG.items():
            # 2 colonnes (noisy, denoised), n_variants lignes
            fig, axes = plt.subplots(n_variants, 2, figsize=(6, 2.5 * n_variants))
            if n_variants == 1:
                axes = np.expand_dims(axes, axis=0)
            for v in range(n_variants):
                # Choix du niveau de bruit
                if 'mean' in config and 'std' in config:
                    level = np.clip(np.random.normal(config['mean'], config['std']), *config['clip'])
                elif noise_type == 'poisson':
                    level = None
                else:
                    level = 0.1
                noisy = add_noise(img, noise_type, level if level is not None else 0.1)
                pred = model.predict(noisy[np.newaxis, ...])[0].astype(np.float32)
                psnr_val = tf.image.psnr(img, pred, max_val=1.0).numpy()
                ssim_val = tf.image.ssim(img, pred, max_val=1.0).numpy()
                # Noisy
                axes[v, 0].imshow(np.clip(noisy, 0, 1))
                axes[v, 0].set_title(f"Noisy\n{noise_type} Var {v+1}")
                axes[v, 0].axis('off')
                # Denoised
                axes[v, 1].imshow(np.clip(pred, 0, 1))
                axes[v, 1].set_title(f"Denoised\nPSNR:{psnr_val:.2f} SSIM:{ssim_val:.3f}")
                axes[v, 1].axis('off')
            plt.suptitle(f"{os.path.basename(img_path)} - {noise_type}", fontsize=14, y=1.02)
            plt.tight_layout()
            plt.show()

# %%
test_model_on_deposit(model, n_variants=3)


# %%
import glob

def denoise_images_in_folder(model, deposit_dir="notebooks/livrable2/tests_deposit"):
    image_paths = glob.glob(os.path.join(deposit_dir, "*"))
    if not image_paths:
        print("Aucune image trouvée dans", deposit_dir)
        return

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Impossible de lire {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE) / 255.0
        img_resized = img_resized.astype(np.float32)
        pred = model.predict(img_resized[np.newaxis, ...])[0].astype(np.float32)
        # Affichage côte à côte
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(np.clip(img_resized, 0, 1))
        axes[0].set_title("Original")
        axes[0].axis('off')
        axes[1].imshow(np.clip(pred, 0, 1))
        axes[1].set_title("Denoised")
        axes[1].axis('off')

        # Affichage des métriques intéressantes
        psnr_val = tf.image.psnr(img_resized, pred, max_val=1.0).numpy()
        ssim_val = tf.image.ssim(img_resized, pred, max_val=1.0).numpy()
        axes[1].set_title(f"Denoised\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.3f}")
        axes[1].axis('off')
        plt.suptitle(os.path.basename(img_path), fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()


# %%

denoise_images_in_folder(model)

# %%
import math

def process_image_by_patches(model, image, patch_size=(256, 256), overlap=32):
    """
    Traite une image de toute taille en la divisant en patchs, puis reconstruit l'image complète.
    
    Args:
        model: modèle de débruitage (ou autre) prenant des images de taille patch_size.
        image: np.ndarray, image d'entrée (H, W, C) en float32, valeurs [0,1].
        patch_size: tuple (h, w), taille des patchs à traiter.
        overlap: int, nombre de pixels de chevauchement entre patchs.
        
    Returns:
        np.ndarray: image reconstruite de la même taille que l'entrée.
    """
    h, w, c = image.shape
    ph, pw = patch_size

    # Calcul du nombre de patchs nécessaires
    n_patches_h = math.ceil((h - overlap) / (ph - overlap))
    n_patches_w = math.ceil((w - overlap) / (pw - overlap))

    # Initialisation de la sortie et du masque de normalisation
    output = np.zeros((h, w, c), dtype=np.float32)
    norm_mask = np.zeros((h, w, c), dtype=np.float32)

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y1 = i * (ph - overlap)
            x1 = j * (pw - overlap)
            y2 = min(y1 + ph, h)
            x2 = min(x1 + pw, w)
            y1 = max(0, y2 - ph)
            x1 = max(0, x2 - pw)

            patch = image[y1:y2, x1:x2, :]
            # Si le patch n'est pas de la bonne taille, on le complète par du padding
            pad_h = ph - patch.shape[0]
            pad_w = pw - patch.shape[1]
            if pad_h > 0 or pad_w > 0:
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            # Prédiction sur le patch
            pred_patch = model.predict(patch[np.newaxis, ...])[0]
            # On enlève le padding si besoin
            pred_patch = pred_patch[:patch.shape[0], :patch.shape[1], :]

            # Ajout du patch à la sortie
            output[y1:y2, x1:x2, :] += pred_patch
            norm_mask[y1:y2, x1:x2, :] += 1.0

    # Normalisation pour gérer les zones de chevauchement
    output = output / np.maximum(norm_mask, 1e-8)
    output = np.clip(output, 0, 1)
    return output

img = cv2.imread("notebooks/livrable2/tests_deposit/noise-reduction-1.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
result = process_image_by_patches(model, img, patch_size=(256,256), overlap=32)
plt.imshow(result)
plt.axis('off')
plt.show()

## Create a destination folder and save the output image
output_dir = "notebooks/livrable2/tests_deposit/tests_processed"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "processed_image.jpg")
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)  # Convertir en BGR pour OpenCV
cv2.imwrite(output_path, (result * 255).astype(np.uint8))
print(f"Image enregistrée à : {output_path}")

# %%
plt.figure(figsize=(12,4))
for i, color in enumerate(['R', 'G', 'B']):
    plt.subplot(1,3,i+1)
    plt.hist(result[...,i].flatten(), bins=50, color=color.lower(), alpha=0.7)
    plt.title(f'Distribution {color}')
plt.tight_layout()
plt.show()

# Vérification des stats
print("Min/max par canal :", result.min(axis=(0,1)), result.max(axis=(0,1)))
print("Moyenne par canal :", result.mean(axis=(0,1)))

# %%
import math

def debug_process_image_by_patches(model, image, patch_size=(256, 256), overlap=32, force_rgb=True, debug_steps=True, max_patches=3):
    """
    Version debug : affiche les étapes du traitement par patchs.
    """
    image = image.astype(np.float32)
    if image.max() > 1.1:
        image = image / 255.0
    if force_rgb:
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
    h, w, c = image.shape
    ph, pw = patch_size

    n_patches_h = math.ceil((h - overlap) / (ph - overlap))
    n_patches_w = math.ceil((w - overlap) / (pw - overlap))

    output = np.zeros((h, w, c), dtype=np.float32)
    norm_mask = np.zeros((h, w), dtype=np.float32)  # 2D mask

    patch_count = 0
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y1 = i * (ph - overlap)
            x1 = j * (pw - overlap)
            y2 = min(y1 + ph, h)
            x2 = min(x1 + pw, w)
            y1 = max(0, y2 - ph)
            x1 = max(0, x2 - pw)

            patch = image[y1:y2, x1:x2, :]
            pad_h = ph - patch.shape[0]
            pad_w = pw - patch.shape[1]
            if pad_h > 0 or pad_w > 0:
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            pred_patch = model.predict(patch[np.newaxis, ...])[0].astype(np.float32)
            pred_patch = pred_patch[:patch.shape[0], :patch.shape[1], :]

            if pred_patch.shape[-1] == 1:
                pred_patch = np.repeat(pred_patch, 3, axis=-1)
            elif pred_patch.shape[-1] == 4:
                pred_patch = pred_patch[..., :3]

            output[y1:y2, x1:x2, :] += pred_patch
            norm_mask[y1:y2, x1:x2] += 1.0  # 2D mask

            # DEBUG : Affichage des patchs et de la reconstruction partielle
            if debug_steps and patch_count < max_patches:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(np.clip(patch, 0, 1))
                axes[0].set_title(f"Patch d'entrée [{y1}:{y2},{x1}:{x2}]")
                axes[0].axis('off')
                axes[1].imshow(np.clip(pred_patch, 0, 1))
                axes[1].set_title("Patch prédit")
                axes[1].axis('off')
                # Affichage de la zone de sortie en cours de remplissage
                temp_out = output.copy()
                temp_out = temp_out / np.clip(norm_mask[..., None], 1e-6, None)
                axes[2].imshow(np.clip(temp_out, 0, 1))
                axes[2].set_title("Image partielle (reconstruction)")
                axes[2].axis('off')
                plt.suptitle(f"Patch {patch_count+1}/{max_patches}")
                plt.show()
                patch_count += 1

    norm_mask = np.clip(norm_mask, 1e-6, None)
    output = output / norm_mask[..., None]  # broadcast 2D mask to 3D
    output = np.clip(output, 0, 1)

    # DEBUG : Affichage de l'image finale
    plt.figure(figsize=(6, 6))
    plt.imshow(output)
    plt.title("Image finale reconstituée")
    plt.axis('off')
    plt.show()

    return output

# Test debug sur une image
img = cv2.imread("notebooks/livrable2/tests_deposit/image.png", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
result = debug_process_image_by_patches(model, img, patch_size=(256,256), overlap=32, debug_steps=True, max_patches=3)


