# Rapport de Projet — Détection de Pneumonie par Imagerie Thoracique

---

## 1. Contexte et Objectif

La pneumonie est une infection pulmonaire qui reste l'une des principales causes de mortalité dans le monde. Le diagnostic repose généralement sur l'interprétation de radiographies thoraciques par un radiologue. Ce projet vise à développer un outil d'aide au diagnostic basé sur l'apprentissage profond, capable d'analyser une radiographie pulmonaire et de fournir :

1. Un **diagnostic binaire** (normal / pneumonie)
2. Une **carte de chaleur** superposée à l'image pour visualiser les zones suspectes
3. Un **rapport PDF** exportable

---

## 2. Modèles Explorés

### 2.1 DenseNet121 et MobileNet (Classification seule)

**Architecture :** Réseaux de neurones convolutifs pré-entraînés sur ImageNet, utilisés pour la classification binaire.

**Avantages :**
- Rapides à entraîner et à inférer
- Bien adaptés aux datasets de petite taille grâce au transfer learning

**Inconvénients :**
- **Classification uniquement** — aucune information spatiale sur la localisation des lésions
- Impossible de savoir _où_ le modèle regarde dans l'image
- Utiles pour un tri binaire, mais insuffisants pour un outil d'aide au diagnostic clinique qui nécessite une interprétation visuelle

**Conclusion :** Abandonnés car la classification seule ne répond pas au besoin d'explicabilité.

### 2.2 YOLOv8 (Détection d'objets)

**Architecture :** You Only Look Once — détection d'objets par boîtes englobantes (bounding boxes). Entraîné sur le dataset RSNA Pneumonia Challenge qui fournit des annotations sous forme de boîtes autour des opacités.

**Problèmes rencontrés :**
- **Précision insuffisante** : perte élevée, difficulté à converger
- **Temps d'entraînement long** : environ 11 millions de paramètres, entrée à 640×640 pixels
- **Confiance « normal »** non apprise : calculée heuristiquement comme `1.0 - confiance_max_des_boîtes`, ce qui n'est pas fiable
- **Grad-CAM fragile** : nécessitait une passe arrière séparée et des hooks sur les gradients, avec des problèmes de buffers détachés sur le modèle YOLO
- **Localisation imprécise** : les boîtes englobantes sont trop grossières pour des lésions diffuses ou subtiles

**Conclusion :** Abandonné au profit d'une approche par segmentation.

### 2.3 U-Net Résiduel (Approche retenue)

**Architecture :** U-Net depth-4 avec blocs résiduels, double entrée (image + données tabulaires), double sortie (masque de segmentation + classification binaire). Implémenté en TensorFlow/Keras.

**Avantages :**
- **Segmentation pixel-level** : chaque pixel est classé, donnant une localisation précise des zones atteintes
- **Classification intégrée** : une tête de classification séparée apprend à distinguer normal/pneumonie, pas de heuristique
- **Le masque EST la carte de chaleur** : pas besoin de Grad-CAM, la sortie de segmentation s'utilise directement comme heatmap
- **Modèle compact** : ~2-5 millions de paramètres, entrée 224×224 → entraînement rapide
- **Double objectif** : la perte IoU pour la segmentation et la perte BCE pondérée pour la classification se renforcent mutuellement

**Résultats :**
| Métrique | Valeur |
|----------|--------|
| IoU (masque) | 0.71 |
| Accuracy (classification) | 0.93 |
| AUROC | 0.98 |
| Précision (pneumonie) | 0.93 |
| Rappel (pneumonie) | 0.89 |
| F1-score (pneumonie) | 0.91 |

---

## 3. Pipeline de Traitement

### 3.1 Dataset

**Source :** RSNA Pneumonia Detection Challenge, via le dataset Kaggle « RSNA Pneumonia Processed Dataset » qui convertit les boîtes englobantes en masques de segmentation.

**Structure :**
- 26 684 radiographies thoraciques (PNG) + 26 684 masques correspondants
- Métadonnées patient : âge, sexe, position (AP/PA), diagnostic (0 = normal, 1 = pneumonie)
- Split : 72% entraînement, 8% validation, 10% test (stratifié)

### 3.2 Prétraitement

**Images :**
- Redimensionnement à 224×224
- Normalisation [0, 1]
- Augmentation : luminosité aléatoire, contraste aléatoire (entraînement uniquement)

**Données tabulaires :**
- Encodage du sexe (M→0, F→1) et de la position (AP→0, PA→1)
- Âge plafonné à 90 ans, normalisé MinMax [0, 1]

**Pipeline tf.data :**
- Chargement asynchrone, prefetch AUTOTUNE
- Batch de 64
- Shuffle + repeat pour l'entraînement

### 3.3 Architecture du Modèle

```
Entrée image (224×224×3)
  └─ Encodeur (4 niveaux : 32→64→128→256 canaux, avec blocs résiduels)
       └─ Goulot (bottleneck) 512 canaux + Dropout 0.3
            └─ Décodeur (4 niveaux, sauts de connexion)
                 └─ Fusion avec données tabulaires (Dense 64 → UpSampling 224×224)
                      ├─ Sortie masque : Conv2D 1×1, sigmoïde → (224×224×1)
                      └─ Sortie classe : GlobalAvgPool → Dense 64 → Dense 1, sigmoïde
```

**Fonctions de perte :**
- Masque : **IoU Loss** — `1 - intersection / union`
- Classification : **Binary Crossentropy pondérée** — pondération des classes inversement proportionnelle à leur fréquence

### 3.4 Entraînement

**Hyperparamètres :**
- Optimiseur Adam (learning rate 1e-3)
- Batch size : 64
- Early stopping (patience 5)
- Réduction du LR (facteur 0.2, patience 2)
- Poids des pertes : masque 0.7, classification 0.3

**Plateforme :** Kaggle T4 GPU (15 Go VRAM)

**Durée :** ~10-15 époques avant early stopping

### 3.5 Export et Inférence

**Deux modèles sauvegardés :**
1. `model_unet_full.keras` — modèle complet (image + données tabulaires)
2. `model_unet_inference.keras` — modèle image uniquement (valeurs tabulaires moyennes intégrées)

**Inférence en production :** Le modèle complet est chargé et encapsulé avec des valeurs tabulaires par défaut (âge médian, sexe féminin, position AP) via une couche Lambda.

---

## 4. Application Web

### Stack technique
- **Backend :** Flask (Python)
- **Inférence :** TensorFlow/Keras
- **Frontend :** HTML / CSS / JavaScript vanilla
- **PDF :** ReportLab

### Fonctionnement
1. L'utilisateur télécharge une radiographie thoracique (JPG/PNG, <10 Mo)
2. Le serveur prétraite l'image et exécute l'inférence U-Net
3. Le masque de segmentation est superposé à l'image originale (fond bleuté + colormap JET + boîtes autour des régions détectées)
4. Le verdict (normal/pneumonie) et le niveau de confiance sont calculés
5. Un rapport PDF peut être généré et téléchargé

### Interface utilisateur
- Design sombre moderne avec drag-and-drop
- Affichage côte à côte : image originale / overlay
- Métriques : confiance, nombre de régions détectées, nom du fichier
- Rapport texte éditable + téléchargement PDF

---

## 5. Résultats et Discussion

### Performances
La U-Net résiduelle atteint des performances solides :
- **Segmentation :** IoU de 0.71, ce qui est bon pour des masques dérivés de boîtes englobantes (bruit inévitable dans les annotations)
- **Classification :** AUROC de 0.98, précision de 93 %, rappel de 89 %
- Le rappel (89 %) est légèrement inférieur à la précision (93 %), ce qui signifie que le modèle rate quelques cas de pneumonie — acceptable pour un outil de screening qui priorise la spécificité

### Interprétabilité
Contrairement aux classifieurs purs (DenseNet, MobileNet) et au détecteur YOLO, la U-Net fournit une **carte de chaleur intrinsèque** : le masque de segmentation montre exactement où le modèle « voit » l'opacité.

### Limitations
1. **Données tabulaires par défaut** : l'application web utilise des valeurs par défaut (âge moyen, sexe féminin, position AP) en l'absence de métadonnées patient. Cela peut légèrement dégrader la précision.
2. **Inférence CPU** : ~1-2 secondes par image sur CPU ; un GPU accélérerait significativement.
3. **Non clinique** : outil de recherche et démonstration uniquement. Ne remplace pas un diagnostic médical professionnel.
4. **Pas de tests automatisés** : la pipeline d'inférence n'est pas couverte par des tests unitaires.

---

## 6. Conclusion

Le passage de la classification seule (DenseNet/MobileNet) à la détection (YOLO) puis à la segmentation (U-Net) a permis d'obtenir un outil plus complet et interprétable. La U-Net résiduelle combine segmentation pixel-level et classification binaire dans un modèle compact et rapide à entraîner. L'application web associée fournit une interface utilisateur intuitive pour l'inférence et la génération de rapports.

**Travaux futurs :**
- Réentraînement sans entrée tabulaire pour simplifier le déploiement
- Conteneurisation (Docker) pour la reproductibilité
- Tests unitaires sur la pipeline d'inférence
- Étalonnage de la confiance (temperature scaling)
- Backbone plus profond (EfficientNet) pour améliorer la segmentation
