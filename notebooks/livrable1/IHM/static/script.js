// Mobile menu toggle
document.getElementById('mobile-menu-button').addEventListener('click', function() {
    const menu = document.getElementById('mobile-menu');
    menu.classList.toggle('hidden');
});

// Simuler le chargement du modèle (à remplacer par une vérification réelle si nécessaire)
setTimeout(() => {
    document.getElementById('status-text').textContent = "Prêt";
    document.getElementById('model-progress').classList.remove('model-loading');
    document.getElementById('model-progress').classList.add('bg-green-500');
    document.getElementById('model-progress').style.width = "100%";
}, 2000);

// Dropzone et gestion du fichier
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const imagePreviewContainer = document.getElementById('image-preview-container');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const processingOverlay = document.getElementById('processing-overlay');
const resultsSection = document.getElementById('results');
const photoProgress = document.getElementById('photo-progress');
const nonPhotoProgress = document.getElementById('non-photo-progress');
const photoPercent = document.getElementById('photo-percent');
const nonPhotoPercent = document.getElementById('non-photo-percent');
const finalResult = document.getElementById('final-result');
const resultIcon = document.getElementById('result-icon');
const resultTitle = document.getElementById('result-title');
const resultDesc = document.getElementById('result-desc');

// Gestion visuelle du dropzone
['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('active');
    });
});

['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('active');
    });
});

dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.match('image.*')) {
        handleFile(file);
    }
});

dropzone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file && file.type.match('image.*')) {
        handleFile(file);
    }
});

clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetFileInput();
});

analyzeBtn.addEventListener('click', analyzeImage);

function handleFile(file) {
    if (file.size > 5 * 1024 * 1024) {
        alert('La taille du fichier ne doit pas dépasser 5MB');
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreviewContainer.classList.remove('hidden');
        analyzeBtn.classList.remove('hidden');
        dropzone.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function resetFileInput() {
    fileInput.value = '';
    imagePreview.src = '#';
    imagePreviewContainer.classList.add('hidden');
    analyzeBtn.classList.add('hidden');
    dropzone.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    finalResult.classList.add('hidden');
}

// Fonction d'animation des valeurs
function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        element.textContent = Math.floor(progress * (end - start) + start) + '%';
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Fonction modifiée qui appelle l'API Flask /predict et met à jour l'IHM
function analyzeImage() {
    processingOverlay.classList.remove('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('opacity-50');
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('image', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        processingOverlay.classList.add('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('opacity-50');

        if (data.error) {
            resultsSection.innerHTML = "<p class='text-red-500 text-center'>" + data.error + "</p>";
            return;
        }

        // Récupération de la prédiction et des probabilités
        const predictedClass = data.predicted_class;
        const probabilities = data.probabilities;
        let photoProb = parseFloat(probabilities["Photo"]) * 100;
        photoProb = Math.round(photoProb);
        let bestNonPhotoProb = 0;
        let bestNonPhotoClass = "";
        for (let key in probabilities) {
            if (key !== "Photo") {
                let prob = parseFloat(probabilities[key]);
                if (prob > bestNonPhotoProb) {
                    bestNonPhotoProb = prob;
                    bestNonPhotoClass = key;
                }
            }
        }
        bestNonPhotoProb = Math.round(bestNonPhotoProb * 100);

        animateValue(photoPercent, 0, photoProb, 1000);
        animateValue(nonPhotoPercent, 0, bestNonPhotoProb, 1000);

        setTimeout(() => {
            photoProgress.style.width = photoProb + "%";
            nonPhotoProgress.style.width = bestNonPhotoProb + "%";

            if (predictedClass === "Photo") {
                resultIcon.className = 'w-10 h-10 rounded-full flex items-center justify-center mr-3 bg-green-500';
                resultIcon.innerHTML = '<i class="fas fa-check text-white"></i>';
                resultTitle.textContent = "Photo réelle détectée";
                resultTitle.className = 'font-bold text-green-500';
                resultDesc.textContent = "Notre modèle est sûr à " + photoProb + "% qu'il s'agit d'une photo réelle.";
            } else {
                resultIcon.className = 'w-10 h-10 rounded-full flex items-center justify-center mr-3 bg-red-500';
                resultIcon.innerHTML = '<i class="fas fa-times text-white"></i>';
                resultTitle.textContent = "Non-photo détectée";
                resultTitle.className = 'font-bold text-red-500';
                resultDesc.textContent = "Notre modèle est sûr à " + bestNonPhotoProb + "% qu'il ne s'agit pas d'une photo réelle.";
            }
            
            finalResult.classList.remove('hidden');
            resultsSection.classList.remove('hidden');
            document.querySelectorAll('.result-card').forEach((card, index) => {
                setTimeout(() => {
                    card.classList.add('show');
                }, index * 200);
            });
        }, 100);
    })
    .catch(err => {
        processingOverlay.classList.add('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('opacity-50');
        resultsSection.innerHTML = "<p class='text-red-500 text-center'>Erreur lors de la prédiction</p>";
        console.error(err);
    });
}