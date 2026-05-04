# Chapitre 3 : Conception

## Introduction

Ce chapitre présente la conception complète du système d'anonymisation d'images médicales. Il détaille l'architecture globale du système, le modèle de données MongoDB, la conception de l'API REST, les mécanismes de sécurité et l'architecture du pipeline d'intelligence artificielle.

## Architecture du système

### Architecture globale en trois couches

L'architecture du système repose sur une organisation en trois couches fonctionnelles distinctes et communicantes, chacune dotée d'une responsabilité clairement délimitée.

![Architecture globale du système en trois couches](example-image)

La **couche de présentation** est assurée par l'application React servie par Vite sur le port 3000. Elle gère l'interface utilisateur, l'authentification côté client (stockage du token JWT dans `localStorage`), la construction des requêtes `multipart/form-data` et la présentation des résultats d'anonymisation. Les pages principales sont la page de connexion, le tableau de bord (upload et historique) et le panneau d'administration.

La **couche applicative** est assurée par le serveur Express (Node.js) sur le port 5000. Elle gère l'authentification et l'autorisation (middleware JWT), la validation des fichiers uploadés (Multer), la persistance des journaux en MongoDB et la délégation du traitement IA au pipeline FastAPI. Cette couche joue le rôle de passerelle sécurisée : l'API FastAPI n'est jamais accessible directement depuis le frontend.

La **couche de traitement IA** est une API FastAPI autonome sur le port 8000. Elle exécute le pipeline d'anonymisation en sept étapes et communique avec MinIO pour le stockage des résultats. Cette séparation garantit que les opérations de deep learning coûteuses en ressources n'impactent pas la disponibilité des routes d'authentification et de consultation.

### Flux de données complet

Le flux de données d'une requête d'anonymisation suit le chemin suivant. L'utilisateur soumet une image via l'interface React avec les paramètres souhaités. La requête est interceptée par le middleware JWT du backend Node.js qui vérifie l'authenticité du token. Après validation, le contrôleur d'anonymisation crée un objet `FormData` intégrant le fichier image et les quatre paramètres du pipeline, puis le transmet à FastAPI. FastAPI exécute les sept étapes du pipeline, stocke le résultat dans MinIO et retourne un JSON contenant les statistiques et l'URL pré-signée. Le backend Node.js persiste le journal de l'opération en MongoDB et transmet la réponse au frontend.

![Diagramme de déploiement du système](example-image)

## Conception de la base de données

### Modèle de données MongoDB

La base de données MongoDB `medical_anonymizer` héberge deux collections principales : `users` et `logs` (désignée `histories` dans l'implémentation).

#### Collection `users`

| Champ | Type Mongoose | Requis | Contraintes / Description |
|-------|---------------|--------|---------------------------|
| `_id` | ObjectId | Auto | Identifiant unique généré par MongoDB |
| `name` | String | Oui | Longueur max 50 caractères |
| `email` | String | Oui | Unique, indexé, validé par regex |
| `password` | String | Oui | Hash bcrypt, `select: false` |
| `role` | String | Oui | Enum : utilisateur / utilisateur_medical / responsable |
| `createdAt` | Date | Auto | Horodatage de création |
| `updatedAt` | Date | Auto | Horodatage de dernière modification |

#### Collection `histories`

| Champ | Type | Requis | Description |
|-------|------|--------|-------------|
| `_id` | ObjectId | Auto | Identifiant unique |
| `user` | ObjectId (ref) | Oui | Référence à l'utilisateur |
| `filename` | String | Oui | Nom du fichier original |
| `status` | String | Oui | `success` ou `failed` |
| `category` | String | Non | Catégorie CLIP (chest, dental, etc.) |
| `confidence` | Number | Non | Score de confiance CLIP (0–1) |
| `regionsDetected` | Number | Non | Nombre de régions OCR détectées |
| `regionsRedacted` | Number | Non | Nombre de régions supprimées |
| `tagsAnonymized` | Number | Non | Nombre de tags DICOM anonymisés |
| `processingTime` | Number | Non | Durée totale de traitement (ms) |
| `minioUri` | String | Non | URI interne MinIO |
| `downloadUrl` | String | Non | URL pré-signée (24h) |
| `confThreshold` | Number | Non | Seuil de confiance OCR utilisé |
| `padding` | Number | Non | Rembourrage en pixels |
| `borderMargin` | Number | Non | Marge de sécurité en pixels |
| `borderPct` | Number | Non | Pourcentage de scan bordural |
| `createdAt` | Date | Auto | Horodatage de l'opération |

### Dictionnaire des données principal

Les champs `email` et `user` (référence) sont indexés pour optimiser les performances des requêtes de consultation d'historique et d'authentification. Le champ `password` est exclu des requêtes par défaut (`select: false`) afin de prévenir toute exposition accidentelle dans les réponses API.

## Conception de l'API REST

### Endpoints du backend Node.js

| Méthode | Route | Protection | Description |
|---------|-------|------------|-------------|
| POST | `/api/auth/register` | Publique | Inscription d'un utilisateur |
| POST | `/api/auth/login` | Publique | Connexion et émission JWT |
| GET | `/api/auth/me` | JWT | Profil de l'utilisateur courant |
| POST | `/api/anonymize` | JWT | Pipeline d'anonymisation complet |
| GET | `/api/history` | JWT | Historique personnel |
| GET | `/api/history/images/all` | JWT + Médical | Toutes les images du système |
| GET | `/api/history/admin/stats` | JWT + Admin | Statistiques globales |
| GET | `/api/history/admin/users` | JWT + Admin | Liste des utilisateurs |
| PUT | `/api/history/admin/users/:id/role` | JWT + Admin | Modification du rôle |
| DELETE | `/api/history/admin/users/:id` | JWT + Admin | Suppression d'un utilisateur |
| GET | `/api/health` | Publique | État du serveur |

### Endpoints de l'API FastAPI

| Méthode | Route | Protection | Description |
|---------|-------|------------|-------------|
| GET | `/` | Publique | Informations sur le service IA |
| GET | `/health` | Publique | État des composants (CLIP, OCR, OpenCV) |
| POST | `/anonymize` | Réseau | Pipeline complet d'anonymisation |
| GET | `/docs` | Publique | Documentation Swagger auto-générée |

### Formats de requête et de réponse

La route `POST /api/anonymize` attend un corps de requête `multipart/form-data` contenant le champ `file` (image binaire) et les quatre paramètres `conf_threshold` (float), `padding` (int), `border_margin` (int) et `border_pct` (float). La réponse en cas de succès est un JSON conforme à la structure suivante :

```json
{
  "status": "success",
  "filename": "anonymized_scan.jpg",
  "category": "chest",
  "confidence": 0.94,
  "regions_detected": 8,
  "regions_redacted": 7,
  "regions_skipped": 1,
  "tags_anonymized": 23,
  "processing_time": 2.34,
  "download_url": "http://localhost:9000/anonymized-images/...",
  "format": "JPEG"
}
```

## Conception de la sécurité

### Authentification et autorisation

#### Flux JWT

L'authentification repose sur des tokens JWT signés avec HMAC-SHA256. Le payload du token contient l'identifiant utilisateur et la date d'expiration (7 jours). Le middleware `protect` vérifie la signature du token, récupère l'utilisateur en base de données et l'attache à l'objet requête. Les middlewares `responsableOnly` et `medicalOrResponsable` contrôlent l'accès aux routes sensibles en fonction du rôle persisté en base de données (et non dans le token), ce qui garantit que les changements de rôle sont immédiatement effectifs.

![Flux d'authentification JWT](example-image)

#### Gestion des mots de passe

Les mots de passe sont hachés avec bcrypt au facteur de coût 12, offrant un équilibre optimal entre sécurité et performance. Le hook Mongoose `pre('save')` déclenche automatiquement le hachage avant toute persistance. La méthode `matchPassword` sur le modèle User encapsule la comparaison sécurisée avec `bcrypt.compare`.

### Gestion des rôles et permissions (RBAC)

| Action | Utilisateur | Util. médical | Responsable |
|--------|-------------|---------------|-------------|
| Uploader et anonymiser | Oui | Oui | Oui |
| Consulter son historique | Oui | Oui | Oui |
| Consulter tous les historiques | Non | Oui | Oui |
| Télécharger toutes les images | Non | Oui | Oui |
| Accéder aux statistiques admin | Non | Non | Oui |
| Gérer les utilisateurs | Non | Non | Oui |
| Modifier les rôles | Non | Non | Oui |
| Supprimer des utilisateurs | Non | Non | Oui |

### Upload sécurisé avec Multer

Le middleware Multer est configuré avec un stockage en mémoire vive (`memoryStorage`), évitant toute écriture de fichiers originaux sur le disque du serveur backend. Le filtre de fichiers (`fileFilter`) valide les extensions autorisées (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.dcm`, `.dicom`) avant d'accepter le fichier. La taille maximale est fixée à 50 Mo.

## Conception du modèle d'IA

### Pipeline d'anonymisation en sept étapes

![Pipeline d'anonymisation en sept étapes](example-image)

Le pipeline d'anonymisation est organisé en sept étapes séquentielles, chacune avec un rôle précis et des mécanismes de gestion des erreurs permettant une dégradation gracieuse en cas d'échec partiel.

#### Étape 1 : Classification (CLIP ViT-B/32)

Le modèle CLIP ViT-B/32 analyse l'image en mode zéro-shot à partir de prompts textuels descriptifs. Cinq catégories médicales sont définies (_chest_, _dental_, _pelvic_, _skull_, _other_medical_) plus une catégorie de rejet (_non_medical_). Le modèle retourne la catégorie de similarité cosinus maximale avec son score de confiance. Si l'image est classifiée comme non médicale ou de catégorie non supportée, le pipeline s'arrête avec un code HTTP 400.

#### Étape 2 : Validation du format

Le module `ImageValidator` vérifie la cohérence du fichier : extension supportée, lisibilité du fichier image ou DICOM, et extraction du tableau de pixels. Pour les fichiers DICOM, cette étape tente de décompresser les données pixel en utilisant les codecs disponibles (`python-gdcm`, `pylibjpeg`).

#### Étape 3 : Anonymisation des métadonnées DICOM

Pour les fichiers DICOM uniquement, le module `MetadataAnonymizer` parcourt les tags PHI identifiés selon les listes HIPAA et les remplace par des valeurs vides ou génériques. Le nombre de tags modifiés est consigné dans le journal de l'opération.

#### Étape 4 : Prétraitement CLAHE

Le module `BorderPreprocessor` applique une amélioration de contraste CLAHE (Contrast Limited Adaptive Histogram Equalization) limitée aux zones bordurales de l'image (les `border_pct` premiers et derniers pourcents des lignes et colonnes). Cette amélioration locale augmente le contraste du texte incrustés sans altérer la zone diagnostique centrale.

#### Étape 5 : Détection OCR double moteur

PaddleOCR PP-OCRv4 est exécuté en premier sur l'image complète prétraitée, avec le seuil de confiance paramétrable. EasyOCR est ensuite exécuté avec le même seuil sur la région bordurale uniquement. Les deux listes de boîtes englobantes sont fusionnées par déduplication IoU : si l'IoU entre une boîte EasyOCR et une boîte PaddleOCR dépasse 0,5, la boîte EasyOCR est considérée comme doublon et ignorée.

#### Étape 6 : Suppression des régions textuelles

Le module `PixelRedactor` applique une logique de sécurité à deux niveaux. Seules les régions situées à moins de `border_margin` pixels des bords de l'image sont automatiquement supprimées (zone de sécurité). Les régions centrales sont journalisées mais préservées, évitant la suppression accidentelle de données diagnostiques. Pour les régions à supprimer, deux stratégies sont appliquées selon le fond détecté :

- **Fond sombre** (luminance moyenne < 60) : remplissage par la médiane des pixels d'un bandeau de 15 pixels prélevé sur le bord le plus proche de l'image.
- **Fond clair** (radiographie typique) : reconstruction par inpainting OpenCV TELEA avec un rayon de 1 pixel, produisant un résultat visuellement naturel.

#### Étape 7 : Sauvegarde et stockage MinIO

L'image traitée est sauvegardée dans le dossier de sortie local, puis uploadée dans MinIO via le client `MinIOStorage`. Les images sont organisées par catégorie anatomique et par date (`catégorie/YYYY/MM/DD/nomfichier`). Une URL pré-signée valable 24 heures est générée et incluse dans la réponse JSON.

### Architecture du classifieur CLIP

![Architecture du classifieur CLIP zéro-shot](example-image)

Le modèle CLIP encode l'image en entrée et les descriptions textuelles des catégories dans un espace vectoriel commun de dimension 512. La similarité cosinus entre l'embedding de l'image et chaque embedding textuel est calculée et normalisée par softmax pour obtenir une distribution de probabilité sur les catégories.

### Métriques d'évaluation du pipeline OCR

L'évaluation des performances du pipeline OCR repose sur les métriques standard de détection : la précision (P), le rappel (R) et le F1-Score, définis comme suit :

**Précision** = TP / (TP + FP)

**Rappel** = TP / (TP + FN)

**F1-Score** = 2 × (Précision × Rappel) / (Précision + Rappel)

où TP est le nombre de régions textuelles correctement détectées, FP le nombre de fausses détections, et FN le nombre de régions manquées.

## Conclusion

Ce chapitre a présenté la conception complète du système, de l'architecture en couches au pipeline d'IA. Les choix de conception reflètent un équilibre entre performance, sécurité et maintenabilité. Le chapitre suivant décrit l'implémentation concrète de chacun de ces composants avec des extraits de code significatifs.

---

# Chapitre 4 : Implémentation

## Introduction

Ce chapitre décrit l'implémentation concrète du système d'anonymisation d'images médicales. Il présente l'environnement de développement, la structure du projet, puis détaille les extraits de code les plus significatifs de chaque module : backend Node.js, pipeline IA FastAPI, stockage MinIO et interface React.

## Environnement de développement

### Outils et technologies

L'environnement de développement repose sur les outils suivants : Visual Studio Code comme éditeur principal, Git et GitHub pour la gestion de versions et la collaboration, et un environnement virtuel Python `venv` pour l'isolation des dépendances. Le développement a été réalisé sur Windows 11 avec Node.js v22.20.0, Python 3.10, MongoDB 8.0 local, et MinIO déployé via Docker.

### Structure complète du projet

```
PFE_Test/
├── api/                        # Pipeline IA FastAPI
│   ├── main.py                 # Point d'entrée + 7 étapes du pipeline
│   ├── config.py               # Configuration MinIO (Pydantic Settings)
│   └── storage.py              # Client MinIO S3-compatible
├── anonymizer/                 # Modules d'anonymisation
│   ├── pixel_redactor.py       # Redaction adaptative OpenCV TELEA
│   ├── metadata_anonymizer.py  # Suppression des tags DICOM PHI
│   ├── image_validator.py      # Validation format + extraction pixels
│   └── dicom_decompressor.py   # Décompression DICOM (gdcm/pylibjpeg)
├── ocr/                        # Moteurs OCR
│   ├── text_detector.py        # PaddleOCR PP-OCRv4
│   ├── easy_text_detector.py   # EasyOCR (région bordur.)
│   └── preprocessor.py         # CLAHE - BorderPreprocessor
├── image_classifier/
│   └── improved_medical_classifier.py  # CLIP ViT-B/32 zero-shot
├── backend/                    # Node.js Express API
│   ├── src/
│   │   ├── app.js              # Configuration Express + CORS + routes
│   │   ├── server.js           # Point d'entrée + connexion MongoDB
│   │   ├── controllers/
│   │   │   ├── authController.js       # Inscription, connexion, profil
│   │   │   ├── anonymizeController.js  # Orchestration pipeline IA
│   │   │   └── historyController.js    # Historique + admin
│   │   ├── middleware/
│   │   │   ├── auth.js         # JWT protect + RBAC middlewares
│   │   │   └── upload.js       # Multer memoryStorage
│   │   ├── models/
│   │   │   ├── User.js         # Schema Mongoose utilisateur
│   │   │   └── History.js      # Schema Mongoose journal
│   │   └── routes/
│   │       ├── auth.js
│   │       ├── anonymize.js
│   │       └── history.js
│   ├── package.json
│   └── .env                    # Variables d'environnement
├── client/                     # React + Vite
│   └── src/
│       ├── App.jsx             # Router + AuthContext provider
│       ├── pages/
│       │   ├── Login.jsx       # Page connexion/inscription
│       │   ├── Dashboard.jsx   # Tableau de bord principal
│       │   └── AdminPanel.jsx  # Panneau administrateur
│       ├── components/
│       │   ├── UploadZone.jsx  # Zone upload + paramètres avancés
│       │   ├── AdvancedSettings.jsx  # Curseurs paramètres pipeline
│       │   └── HistoryList.jsx # Historique des opérations
│       └── context/
│           └── AuthContext.jsx # Contexte auth global
├── requirements.txt            # Dépendances Python fixées
├── docker-compose.yml          # MinIO + services conteneurisés
└── .gitignore
```

## Implémentation du backend Node.js

### Configuration de l'application Express

```javascript
// backend/src/app.js
const express = require('express')
const cors = require('cors')
const morgan = require('morgan')
const app = express()

app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
}))
app.use(express.json({ limit: '10mb' }))
app.use(express.urlencoded({ extended: true }))
app.use(morgan('combined'))

// Routes
app.use('/api/auth',      require('./routes/auth'))
app.use('/api/anonymize', require('./routes/anonymize'))
app.use('/api/history',   require('./routes/history'))
app.use('/api/health', (req, res) =>
  res.json({ status: 'ok', timestamp: new Date() }))

// Gestion globale des erreurs
app.use((err, req, res, next) => {
  console.error(err.stack)
  res.status(err.statusCode || 500).json({
    error: err.message || 'Erreur serveur interne'
  })
})

module.exports = app
```

### Middleware d'authentification JWT

```javascript
// backend/src/middleware/auth.js
const jwt = require('jsonwebtoken')
const User = require('../models/User')

exports.protect = async (req, res, next) => {
  let token
  if (req.headers.authorization?.startsWith('Bearer')) {
    token = req.headers.authorization.split(' ')[1]
  }

  if (!token) {
    return res.status(401).json({ error: 'Non autorisé - Token manquant' })
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET)
    req.user = await User.findById(decoded.id).select('-password')
    next()
  } catch (error) {
    return res.status(401).json({ error: 'Token invalide ou expiré' })
  }
}

exports.responsableOnly = (req, res, next) => {
  if (req.user && req.user.role === 'responsable') {
    next()
  } else {
    res.status(403).json({ error: 'Accès refusé - Rôle responsable requis' })
  }
}
```

### Contrôleur d'anonymisation

```javascript
// backend/src/controllers/anonymizeController.js
const FormData = require('form-data')
const axios = require('axios')
const History = require('../models/History')

exports.anonymizeImage = async (req, res) => {
  try {
    const { conf_threshold, padding, border_margin, border_pct } = req.body
    
    const formData = new FormData()
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    })
    formData.append('conf_threshold', conf_threshold || '0.1')
    formData.append('padding', padding || '5')
    formData.append('border_margin', border_margin || '150')
    formData.append('border_pct', border_pct || '0.15')

    const response = await axios.post(
      `${process.env.FASTAPI_URL}/anonymize`,
      formData,
      { headers: formData.getHeaders(), timeout: 30000 }
    )

    const historyEntry = await History.create({
      user: req.user._id,
      filename: req.file.originalname,
      status: 'success',
      category: response.data.category,
      confidence: response.data.confidence,
      regionsDetected: response.data.regions_detected,
      regionsRedacted: response.data.regions_redacted,
      tagsAnonymized: response.data.tags_anonymized || 0,
      processingTime: response.data.processing_time,
      downloadUrl: response.data.download_url,
      confThreshold: parseFloat(conf_threshold),
      padding: parseInt(padding),
      borderMargin: parseInt(border_margin),
      borderPct: parseFloat(border_pct)
    })

    res.status(200).json(response.data)
  } catch (error) {
    await History.create({
      user: req.user._id,
      filename: req.file?.originalname || 'unknown',
      status: 'failed'
    })
    res.status(500).json({ error: error.message })
  }
}
```

## Implémentation du pipeline FastAPI

### Point d'entrée principal

```python
# api/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import time

app = FastAPI(title="Medical Image Anonymization API")

@app.post("/anonymize")
async def anonymize_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.1),
    padding: int = Form(5),
    border_margin: int = Form(150),
    border_pct: float = Form(0.15)
):
    start_time = time.time()
    
    try:
        # Étape 1: Classification CLIP
        classifier = ImprovedMedicalClassifier()
        image_bytes = await file.read()
        category, confidence = classifier.classify_image(image_bytes)
        
        if category == "non_medical":
            raise HTTPException(400, "Image non médicale rejetée")
        
        # Étape 2: Validation format
        validator = ImageValidator()
        img_array, is_dicom = validator.validate_and_extract(
            image_bytes, file.filename
        )
        
        # Étape 3: Anonymisation DICOM
        tags_anonymized = 0
        if is_dicom:
            anonymizer = MetadataAnonymizer()
            tags_anonymized = anonymizer.anonymize_dicom(image_bytes)
        
        # Étape 4: Prétraitement CLAHE
        preprocessor = BorderPreprocessor()
        preprocessed = preprocessor.enhance_borders(img_array, border_pct)
        
        # Étape 5: Détection OCR double moteur
        paddle_detector = TextDetector(conf_threshold=conf_threshold)
        easy_detector = EasyTextDetector(
            conf_threshold=conf_threshold,
            border_pct=border_pct
        )
        
        paddle_boxes = paddle_detector.detect(preprocessed)
        easy_boxes = easy_detector.detect_border_only(
            preprocessed, border_pct
        )
        
        # Fusion IoU
        all_boxes = merge_boxes_iou(paddle_boxes, easy_boxes, threshold=0.5)
        
        # Étape 6: Redaction adaptative
        redactor = PixelRedactor(
            padding=padding,
            border_margin=border_margin
        )
        anonymized, redacted_count, skipped_count = redactor.redact_regions(
            img_array, all_boxes
        )
        
        # Étape 7: Sauvegarde MinIO
        output_filename = f"anonymized_{int(time.time())}_{file.filename}"
        storage = MinIOStorage()
        download_url = storage.upload_image(
            anonymized, output_filename, category
        )
        
        processing_time = time.time() - start_time
        
        return JSONResponse({
            "status": "success",
            "filename": output_filename,
            "category": category,
            "confidence": float(confidence),
            "regions_detected": len(all_boxes),
            "regions_redacted": redacted_count,
            "regions_skipped": skipped_count,
            "tags_anonymized": tags_anonymized,
            "processing_time": round(processing_time, 2),
            "download_url": download_url
        })
        
    except Exception as e:
        raise HTTPException(500, str(e))
```

### Module de redaction adaptative

```python
# anonymizer/pixel_redactor.py
import cv2
import numpy as np

class PixelRedactor:
    def __init__(self, padding=5, border_margin=150):
        self.padding = padding
        self.border_margin = border_margin
    
    def redact_regions(self, image, boxes):
        redacted_count = 0
        skipped_count = 0
        h, w = image.shape[:2]
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Zone de sécurité : bordures uniquement
            is_safe = (x1 < self.border_margin or 
                      x2 > w - self.border_margin or
                      y1 < self.border_margin or 
                      y2 > h - self.border_margin)
            
            if not is_safe:
                skipped_count += 1
                continue
            
            # Expansion avec padding
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(w, x2 + self.padding)
            y2 = min(h, y2 + self.padding)
            
            # Détection fond sombre vs clair
            region = image[y1:y2, x1:x2]
            mean_intensity = np.mean(region)
            
            if mean_intensity < 60:  # Fond sombre
                # Échantillonnage couleur de fond
                border_sample = self._sample_border_color(
                    image, x1, y1, x2, y2
                )
                image[y1:y2, x1:x2] = border_sample
            else:  # Fond clair - inpainting
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                image = cv2.inpaint(
                    image, mask, inpaintRadius=1, 
                    flags=cv2.INPAINT_TELEA
                )
            
            redacted_count += 1
        
        return image, redacted_count, skipped_count
    
    def _sample_border_color(self, image, x1, y1, x2, y2):
        h, w = image.shape[:2]
        band_width = 15
        
        # Prélèvement bande du bord le plus proche
        if y1 < h // 2:
            band = image[0:band_width, x1:x2]
        else:
            band = image[h-band_width:h, x1:x2]
        
        return np.median(band, axis=(0, 1)).astype(np.uint8)
```

## Implémentation du frontend React

### Composant UploadZone avec paramètres avancés

```jsx
// client/src/components/UploadZone.jsx
import { useState } from 'react'
import AdvancedSettings from './AdvancedSettings'
import axios from 'axios'

export default function UploadZone({ onSuccess }) {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [params, setParams] = useState({
    conf_threshold: 0.1,
    padding: 5,
    border_margin: 150,
    border_pct: 0.15
  })

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) return

    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)
    formData.append('conf_threshold', params.conf_threshold)
    formData.append('padding', params.padding)
    formData.append('border_margin', params.border_margin)
    formData.append('border_pct', params.border_pct)

    try {
      const token = localStorage.getItem('token')
      const response = await axios.post(
        'http://localhost:5000/api/anonymize',
        formData,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'multipart/form-data'
          }
        }
      )
      onSuccess(response.data)
    } catch (error) {
      alert('Erreur: ' + error.response?.data?.error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="upload-zone">
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept=".jpg,.jpeg,.png,.dcm,.dicom"
          onChange={(e) => setFile(e.target.files[0])}
        />
        
        <AdvancedSettings params={params} setParams={setParams} />
        
        <button type="submit" disabled={!file || loading}>
          {loading ? 'Traitement...' : 'Anonymiser'}
        </button>
      </form>
    </div>
  )
}
```

## Conclusion

Ce chapitre a présenté l'implémentation concrète des composants principaux du système. Les extraits de code illustrent les choix techniques et les bonnes pratiques adoptées. Le chapitre suivant présente la stratégie de tests et les résultats de validation obtenus.

---

_(Continuez avec les chapitres 5, 6, conclusion et annexes dans le fichier RAPPORT_PFE_COMPLET.md)_
