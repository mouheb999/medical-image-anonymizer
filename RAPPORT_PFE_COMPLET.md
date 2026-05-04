# RAPPORT DE PROJET DE FIN D'ÉTUDES

---

**RÉPUBLIQUE TUNISIENNE**  
MINISTÈRE DE L'ENSEIGNEMENT SUPÉRIEUR ET DE LA RECHERCHE SCIENTIFIQUE

**INSTITUT SUPÉRIEUR D'INFORMATIQUE ET DES TECHNIQUES  
DE COMMUNICATION DE HAMMAM SOUSSE (ISITCOM)**

---

## PROJET DE FIN D'ÉTUDES

Pour l'obtention du diplôme de :  
**Licence Appliquée en Informatique**

---

### Développement d'un modèle d'intelligence artificielle pour l'anonymisation et le stockage sécurisé d'images médicales

---

**Réalisé par :** Mouheb Boubrik

**Encadreur académique :** M. Mehdi Azzouzi, Enseignant — ISITCOM  
**Encadreur professionnel :** M. Mohamed Sbika, Backend Engineer — Pura Solutions

**Année universitaire : 2025–2026**

---

# Dédicaces

> _À mes parents,_  
> _pour leur amour inconditionnel, leurs sacrifices et leur soutien constant._
>
> _À mes frères et sœurs,_  
> _pour leur présence et leurs encouragements._
>
> _À tous ceux qui ont cru en moi,_  
> _ce travail est le fruit de votre confiance._

---

# Remerciements

Je tiens à exprimer ma profonde gratitude à toutes les personnes qui ont contribué, de près ou de loin, à la réalisation de ce projet de fin d'études.

Mes remerciements s'adressent en premier lieu à **M. Mehdi Azzouzi**, mon encadreur académique à l'ISITCOM, pour ses conseils avisés, sa disponibilité et son soutien tout au long de ce projet.

Je remercie également **M. Mohamed Sbika**, mon encadreur professionnel chez Pura Solutions, pour son expertise technique, ses orientations précieuses et la confiance qu'il m'a accordée.

Mes remerciements vont aussi à l'ensemble du corps enseignant de l'ISITCOM pour la qualité de la formation dispensée, ainsi qu'à l'équipe de Pura Solutions pour leur accueil chaleureux.

Enfin, je remercie ma famille et mes amis pour leur soutien moral et leurs encouragements constants.

---

# Résumé

## Résumé en français

**Contexte :** La protection des données personnelles de santé est devenue un enjeu majeur avec l'entrée en vigueur du RGPD et des réglementations internationales telles que HIPAA. Les images médicales contiennent fréquemment des informations identifiantes sous forme de métadonnées DICOM et de texte incrusté dans les pixels, exposant les établissements de santé à des risques juridiques et réputationnels en cas de divulgation non maîtrisée.

**Objectif :** Ce projet vise à développer une plateforme web complète pour l'anonymisation automatique et sécurisée d'images médicales, basée sur un pipeline d'intelligence artificielle en sept étapes.

**Solution :** Le système développé intègre CLIP ViT-B/32 pour la classification zéro-shot des images en catégories anatomiques, un double moteur OCR (PaddleOCR PP-OCRv4 + EasyOCR) avec fusion par IoU pour la détection exhaustive du texte incrusté, et un module de suppression adaptative par inpainting OpenCV. L'application web MERN (MongoDB, Express, React, Node.js) sécurisée par JWT et RBAC permet aux utilisateurs de soumettre des images, de paramétrer le pipeline et de télécharger les résultats anonymisés stockés dans MinIO.

**Résultats :** Le pipeline OCR double moteur atteint un F1-Score de 93,4% avec un rappel de 93,1%, garantissant la quasi-exhaustivité des détections. Le classifieur CLIP atteint un taux de classification correct supérieur à 90% sur toutes les catégories médicales testées. Le temps de traitement moyen est de 2,8 secondes pour une image JPEG standard sur CPU.

**Mots-clés :** Intelligence Artificielle, Anonymisation, Images Médicales, OCR, Deep Learning, CLIP, PaddleOCR, MERN, RGPD, Sécurité des Données.

---

## Abstract (English)

**Context:** The protection of personal health data has become a major issue with the entry into force of the GDPR and international regulations such as HIPAA. Medical images frequently contain identifying information in the form of DICOM metadata and text embedded in pixels, exposing healthcare facilities to legal and reputational risks in the event of uncontrolled disclosure.

**Objective:** This project aims to develop a complete web platform for the automatic and secure anonymization of medical images, based on a seven-stage artificial intelligence pipeline.

**Solution:** The developed system integrates CLIP ViT-B/32 for zero-shot classification of images into anatomical categories, a dual OCR engine (PaddleOCR PP-OCRv4 + EasyOCR) with IoU fusion for exhaustive detection of embedded text, and an adaptive removal module using OpenCV inpainting. The MERN web application (MongoDB, Express, React, Node.js) secured by JWT and RBAC allows users to submit images, configure the pipeline, and download anonymized results stored in MinIO.

**Results:** The dual OCR pipeline achieves an F1-Score of 93.4% with a recall of 93.1%, ensuring near-exhaustive detection. The CLIP classifier achieves a correct classification rate above 90% on all tested medical categories. The average processing time is 2.8 seconds for a standard JPEG image on CPU.

**Keywords:** Artificial Intelligence, Anonymization, Medical Images, OCR, Deep Learning, CLIP, PaddleOCR, MERN, GDPR, Data Security.

---

## Résumé en arabe (ملخص)

_Note : Le résumé en langue arabe est à insérer ici en utilisant un éditeur compatible (Microsoft Word, ou en ajoutant le package polyglossia avec la configuration de la langue arabe dans le préambule Overleaf)._

**Contenu à traduire :** Ce projet développe une plateforme web complète pour l'anonymisation automatique d'images médicales, basée sur un pipeline IA en sept étapes intégrant CLIP ViT-B/32 pour la classification zéro-shot, un double moteur OCR (PaddleOCR PP-OCRv4 + EasyOCR) avec fusion IoU, et un module de suppression adaptative par inpainting OpenCV. L'application MERN sécurisée par JWT et RBAC stocke les images anonymisées dans MinIO. Le F1-Score du pipeline OCR atteint 93,4% avec préservation complète de l'intégrité diagnostique.

**Mots-clés (AR) :** Intelligence Artificielle, Anonymisation, Images Médicales, Apprentissage Profond, Sécurité des Données, RGPD.

---

# Liste des abréviations

| Abréviation | Signification |
|-------------|---------------|
| API | Application Programming Interface |
| CLAHE | Contrast Limited Adaptive Histogram Equalization |
| CLIP | Contrastive Language–Image Pre-training |
| CNN | Convolutional Neural Network |
| CORS | Cross-Origin Resource Sharing |
| CRNN | Convolutional Recurrent Neural Network |
| DB-Net | Differentiable Binarization Network |
| DICOM | Digital Imaging and Communications in Medicine |
| GDPR | General Data Protection Regulation |
| HMAC | Hash-based Message Authentication Code |
| HTTP | HyperText Transfer Protocol |
| HTTPS | HyperText Transfer Protocol Secure |
| IA | Intelligence Artificielle |
| IoU | Intersection over Union |
| IRM | Imagerie par Résonance Magnétique |
| JWT | JSON Web Token |
| MERN | MongoDB, Express, React, Node.js |
| MinIO | Minimal Object Storage (compatible S3) |
| ML | Machine Learning |
| MongoDB | Base de données NoSQL orientée documents |
| OCR | Optical Character Recognition |
| PACS | Picture Archiving and Communication System |
| PHI | Protected Health Information |
| RBAC | Role-Based Access Control |
| REST | Representational State Transfer |
| RGPD | Règlement Général sur la Protection des Données |
| S3 | Simple Storage Service |
| SSL | Secure Sockets Layer |
| TELEA | Alexandru Telea (algorithme d'inpainting OpenCV) |
| TLS | Transport Layer Security |
| UI | User Interface |
| ViT | Vision Transformer |

---

# Introduction générale

## Contexte général

La transformation numérique du secteur médical s'accompagne d'une production massive de données de santé, dont les images médicales constituent une part significative. Ces images, qu'elles proviennent de radiographies, de scanners, d'IRM ou d'échographies, contiennent des informations diagnostiques essentielles mais également des données personnelles identifiantes qui exposent les établissements de santé à des risques juridiques et éthiques majeurs.

L'entrée en vigueur du Règlement Général sur la Protection des Données (RGPD) en 2018 a renforcé les obligations des acteurs du secteur médical en matière de protection des données à caractère personnel. Les images médicales, lorsqu'elles contiennent des informations permettant d'identifier directement ou indirectement un patient, sont soumises aux exigences strictes du RGPD. Les sanctions en cas de non-conformité peuvent atteindre 4% du chiffre d'affaires mondial annuel ou 20 millions d'euros.

Au-delà des métadonnées DICOM (nom du patient, date de naissance, identifiant, etc.), les images médicales contiennent fréquemment du texte incrusté directement dans les pixels : annotations manuscrites, identifiants de l'établissement, dates d'acquisition, paramètres d'imagerie. Ces informations, invisibles aux outils d'anonymisation basiques qui ne traitent que les métadonnées, constituent une faille majeure dans les processus actuels d'anonymisation.

## Problématique

Les approches traditionnelles d'anonymisation d'images médicales présentent plusieurs limitations fondamentales :

- **Traitement manuel chronophage** : La suppression manuelle des annotations textuelles est fastidieuse, sujette aux erreurs humaines et incompatible avec les volumes croissants d'images à traiter.
- **Solutions partielles** : Les outils existants se concentrent sur les métadonnées DICOM mais ignorent le texte incrusté dans les pixels.
- **Manque d'automatisation** : L'absence de classification automatique des images oblige les utilisateurs à configurer manuellement les paramètres pour chaque type d'image.
- **Absence de traçabilité** : Les processus actuels ne garantissent pas une journalisation systématique des opérations effectuées, compromettant la conformité RGPD.

## Objectifs du projet

L'objectif général de ce projet est de concevoir et développer une plateforme web complète pour l'anonymisation automatique et sécurisée d'images médicales, basée sur un pipeline d'intelligence artificielle en sept étapes.

Les objectifs spécifiques sont les suivants :

1. **Développer un classifieur zéro-shot** basé sur CLIP ViT-B/32 pour la classification automatique des images médicales en catégories anatomiques sans nécessiter de données d'entraînement annotées.

2. **Concevoir un pipeline OCR double moteur** combinant PaddleOCR PP-OCRv4 et EasyOCR avec fusion par IoU pour maximiser le taux de détection du texte incrusté.

3. **Implémenter un module de suppression adaptative** utilisant l'inpainting OpenCV avec échantillonnage de la couleur de fond réelle pour produire des résultats visuellement naturels.

4. **Développer une application web sécurisée** avec la pile MERN (MongoDB, Express, React, Node.js) intégrant l'authentification JWT, le contrôle d'accès RBAC et la journalisation complète des opérations.

5. **Assurer le stockage sécurisé** des images anonymisées dans MinIO avec génération d'URLs pré-signées à durée limitée.

## Méthodologie

La méthodologie adoptée suit une approche itérative en cinq phases :

1. **Analyse et étude de l'art** (2 semaines) : Étude des réglementations (RGPD, HIPAA), analyse des solutions existantes, identification des technologies pertinentes.

2. **Mise en place de l'architecture** (1 semaine) : Configuration de l'environnement de développement, installation des dépendances, mise en place de la structure du projet.

3. **Développement du pipeline IA** (3 semaines) : Implémentation du classifieur CLIP, intégration des moteurs OCR, développement du module de redaction adaptative.

4. **Développement de l'application web** (3 semaines) : Création de l'API REST Node.js, développement de l'interface React, intégration avec le pipeline IA.

5. **Tests, intégration et documentation** (2 semaines) : Tests unitaires, tests d'intégration, tests de sécurité, rédaction de la documentation.

## Structure du rapport

Ce rapport est organisé en six chapitres :

- **Chapitre 1** présente le contexte médical et réglementaire ainsi que l'état de l'art des techniques d'anonymisation.
- **Chapitre 2** détaille l'analyse et la spécification des besoins fonctionnels et non fonctionnels du système.
- **Chapitre 3** expose la conception complète du système, de l'architecture globale au pipeline d'IA.
- **Chapitre 4** décrit l'implémentation concrète avec des extraits de code significatifs.
- **Chapitre 5** présente la stratégie de tests et les résultats de validation obtenus.
- **Chapitre 6** discute les résultats, les contributions, les limitations et les perspectives d'amélioration.

---

# Chapitre 1 : Contexte et état de l'art

## Introduction

Ce chapitre présente le contexte médical et réglementaire dans lequel s'inscrit ce projet, puis expose l'état de l'art des techniques d'anonymisation d'images médicales. Il identifie les limitations des approches existantes et justifie les choix technologiques adoptés pour ce projet.

## Contexte médical et réglementaire

### Le cadre réglementaire de la protection des données de santé

#### Le RGPD (Règlement Général sur la Protection des Données)

Le Règlement Général sur la Protection des Données (RGPD), entré en vigueur en mai 2018 dans l'Union Européenne, constitue le cadre réglementaire de référence en matière de protection des données à caractère personnel. Il définit les données de santé comme une catégorie particulière de données sensibles dont le traitement est soumis à des conditions strictes. Le RGPD impose notamment le principe de minimisation des données, le droit à l'effacement et le principe de protection des données dès la conception (_privacy by design_). Dans le contexte de l'imagerie médicale, toute image contenant des informations permettant d'identifier un patient constitue une donnée à caractère personnel soumise aux exigences du RGPD. L'anonymisation, lorsqu'elle est effectuée de manière irréversible, permet de s'affranchir du champ d'application du règlement pour les usages secondaires tels que la recherche.

#### HIPAA (Health Insurance Portability and Accountability Act)

La loi américaine HIPAA de 1996 définit les informations de santé protégées (_Protected Health Information_, PHI) et impose des obligations strictes quant à leur traitement, stockage et transmission. Bien que d'application territoriale américaine, HIPAA constitue une référence internationale en matière de sécurité des données de santé. Elle identifie dix-huit types d'identifiants à supprimer pour atteindre la désidentification sûre d'un dossier médical, dont le nom, la date de naissance, les numéros d'identification et les données géographiques. Ce projet s'inspire des exigences de HIPAA pour définir les tags PHI à anonymiser dans les fichiers DICOM.

#### Cadre réglementaire tunisien

En Tunisie, la loi organique n° 2004-63 du 27 juillet 2004 portant sur la protection des données à caractère personnel encadre le traitement des données personnelles et prévoit des dispositions spécifiques pour les données médicales. La Commission Nationale de Protection des Données (INPDP) est l'autorité de contrôle compétente pour veiller au respect de ces dispositions. Ce cadre national, combiné aux exigences du RGPD pour les projets à vocation européenne, définit les obligations auxquelles se conforme la solution développée dans ce projet.

### Les risques liés aux données non anonymisées

La divulgation non maîtrisée de données médicales identifiantes expose les organisations à des risques multidimensionnels. Sur le plan juridique, les violations du RGPD peuvent entraîner des amendes pouvant atteindre 4% du chiffre d'affaires mondial annuel ou 20 millions d'euros. Sur le plan réputationnel, les incidents de sécurité impliquant des données de santé génèrent une défiance durable des patients et des partenaires institutionnels. Sur le plan opérationnel enfin, la réidentification de patients à partir d'images non anonymisées peut compromettre des études cliniques entières et invalider des résultats de recherche.

## État de l'art : Anonymisation d'images médicales

### Définitions et concepts fondamentaux

#### Anonymisation vs Pseudonymisation

L'anonymisation est un processus irréversible consistant à rendre impossible toute identification d'une personne à partir de données personnelles. La pseudonymisation, en revanche, est un processus réversible où les données identifiantes sont remplacées par des pseudonymes, permettant une réidentification ultérieure si nécessaire. Le RGPD distingue clairement ces deux concepts : les données anonymisées ne sont plus soumises au règlement, tandis que les données pseudonymisées restent des données à caractère personnel.

#### Le standard DICOM

DICOM (Digital Imaging and Communications in Medicine) est le standard international pour le stockage, l'échange et la transmission d'images médicales numériques. Un fichier DICOM contient à la fois l'image pixel et des métadonnées structurées sous forme de tags. Ces métadonnées incluent des informations techniques (paramètres d'acquisition, modalité) et des informations identifiantes (nom du patient, date de naissance, identifiant). L'anonymisation DICOM consiste à supprimer ou remplacer les tags PHI tout en préservant les informations techniques nécessaires au diagnostic.

### Approches traditionnelles d'anonymisation

#### Masquage de zones fixes

L'approche la plus simple consiste à masquer systématiquement des zones prédéfinies de l'image (coins supérieurs, bandes latérales) où les annotations sont généralement positionnées. Cette méthode, implémentée dans certains logiciels PACS, présente l'avantage de la simplicité mais souffre de plusieurs limitations : elle ne s'adapte pas aux variations de positionnement des annotations selon les équipements, elle peut supprimer des zones diagnostiques utiles, et elle ne détecte pas les annotations positionnées de manière non standard.

#### Suppression des métadonnées DICOM uniquement

De nombreux outils open-source (DICOM Anonymizer, GDCM, dcm4che) se concentrent exclusivement sur la suppression des tags DICOM identifiants. Cette approche, bien que nécessaire, est insuffisante car elle ignore totalement le texte incrusté dans les pixels de l'image. Les études montrent que jusqu'à 40% des images médicales contiennent du texte pixel non couvert par les métadonnées DICOM.

#### Limitations des approches traditionnelles

Les approches traditionnelles partagent plusieurs limitations fondamentales : un faible degré d'automatisation, une forte dépendance à la configuration des équipements, une incapacité à s'adapter à de nouveaux formats ou constructeurs sans reconfiguration manuelle, et une impossibilité de traiter de manière fiable les annotations positionnées de manière non standard.

### Approches basées sur l'intelligence artificielle

#### OCR (Optical Character Recognition) pour la détection de texte

Les moteurs OCR modernes, basés sur des architectures de deep learning (CRNN, DB-Net, Transformer), offrent des performances de détection de texte nettement supérieures aux approches traditionnelles. PaddleOCR, développé par Baidu, utilise une architecture PP-OCRv4 combinant un détecteur DB-Net optimisé et un reconnaisseur CRNN. EasyOCR, basé sur CRAFT pour la détection et un Transformer pour la reconnaissance, excelle dans la détection de texte de petite taille et sur fonds complexes. La combinaison de plusieurs moteurs OCR avec fusion des résultats par IoU (Intersection over Union) permet de maximiser le taux de détection tout en minimisant les faux positifs.

#### Classification d'images par deep learning

Les modèles de classification d'images basés sur des CNN (ResNet, EfficientNet, Vision Transformer) permettent de catégoriser automatiquement les images médicales selon leur modalité ou leur région anatomique. CLIP (Contrastive Language–Image Pre-training), développé par OpenAI, introduit une approche zéro-shot où le modèle peut classifier des images en fonction de descriptions textuelles sans nécessiter de données d'entraînement annotées spécifiques. Cette capacité est particulièrement pertinente pour l'anonymisation d'images médicales où les catégories peuvent varier selon les établissements.

#### Inpainting pour la suppression de régions

L'inpainting est une technique de reconstruction numérique consistant à remplir des régions masquées d'une image en propageant les valeurs des pixels environnants. Les algorithmes classiques (TELEA, Navier-Stokes) produisent des résultats satisfaisants pour les images médicales à fond uniforme. Les approches par deep learning (GAN, diffusion models) offrent des résultats visuellement plus naturels mais nécessitent des ressources de calcul importantes et des données d'entraînement spécifiques.

## Contribution de ce projet

Ce projet propose une approche intégrée combinant les avantages des techniques d'IA modernes tout en évitant leurs limitations respectives :

1. **Classification zéro-shot par CLIP** : Élimination du besoin de données d'entraînement annotées, adaptabilité immédiate à de nouvelles catégories d'images.

2. **Double moteur OCR avec fusion IoU** : Maximisation du taux de détection par complémentarité des moteurs, réduction des faux négatifs.

3. **Redaction adaptative par inpainting** : Échantillonnage de la couleur de fond réelle pour produire des résultats visuellement naturels sans nécessiter de modèles génératifs coûteux.

4. **Architecture web complète** : Plateforme utilisable immédiatement en environnement hospitalier avec authentification, journalisation et stockage sécurisé.

5. **Paramétrage dynamique** : Contrôle fin du pipeline depuis l'interface utilisateur sans nécessiter de compétences techniques.

## Conclusion

Ce chapitre a présenté le contexte médical et réglementaire de l'anonymisation d'images médicales, ainsi que l'état de l'art des techniques existantes. Les limitations des approches traditionnelles justifient le recours à des techniques d'intelligence artificielle modernes. Le chapitre suivant présente l'analyse détaillée des besoins du système à développer.

---

# Chapitre 2 : Analyse et spécification des besoins

## Introduction

Ce chapitre présente l'analyse approfondie des besoins du système d'anonymisation d'images médicales. Il identifie les parties prenantes, décrit les besoins fonctionnels et non fonctionnels, formalise la spécification à travers des cas d'utilisation, et détaille les choix technologiques et la planification du projet.

## Analyse des besoins

### Identification des parties prenantes

#### Les utilisateurs finaux

Les utilisateurs finaux de la plateforme sont les professionnels médicaux qui soumettent des images à anonymiser. Ils comprennent les médecins généralistes et spécialistes, les radiologues et techniciens en imagerie médicale, et les chercheurs médicaux ayant besoin d'accéder à des images dépersonnalisées. Dans la nomenclature du système, ce profil correspond au rôle `utilisateur` (accès à ses propres images) et `utilisateur_medical` (accès aux images de tous les utilisateurs). Ces acteurs se caractérisent par une maîtrise variable des outils informatiques et une priorité accordée à la simplicité d'utilisation et à la rapidité du traitement.

#### Les administrateurs système

Les responsables informatiques et médicaux de l'établissement constituent le profil administrateur (`responsable`) de la plateforme. Ils ont accès aux statistiques d'utilisation, à la gestion des comptes utilisateurs, à la consultation des journaux d'audit et aux paramètres système. Ce rôle est soumis à des contrôles d'accès renforcés, notamment l'obligation de fournir une clé d'enregistrement administrative lors de la création du compte.

#### L'équipe technique

Pura Solutions, en tant qu'intégrateur de la solution, constitue une partie prenante technique majeure. Les exigences de maintenabilité, de modularité du code et de documentation ont été définies en concertation avec les ingénieurs de l'entreprise, notamment M. Mohamed Sbika, encadreur professionnel de ce projet.

### Étude de l'existant et problèmes identifiés

L'analyse des processus actuels d'anonymisation dans les établissements de santé révèle plusieurs problèmes récurrents. Le processus est principalement manuel et repose sur la bonne volonté et la vigilance du personnel. Il n'existe pas de traçabilité systématique des opérations effectuées. Les solutions partielles utilisées (scripts Python ad hoc, outils DICOM basiques) ne couvrent pas le texte incrusté dans les pixels. L'absence d'interface utilisateur unifiée oblige les équipes à utiliser des outils disparates, augmentant le risque d'erreurs et de pertes de données.

### Besoins fonctionnels

#### Module d'authentification et gestion des utilisateurs

Le système doit permettre l'inscription et l'authentification sécurisée des utilisateurs avec trois rôles distincts : `utilisateur`, `utilisateur_medical` et `responsable`. L'authentification repose sur un mécanisme JWT avec une durée de validité de 7 jours. La création d'un compte administrateur nécessite la fourniture d'une clé d'enregistrement spéciale. Le système doit permettre la récupération du profil utilisateur et la gestion des rôles par les administrateurs.

#### Module de téléchargement et traitement d'images

Le système doit accepter le téléchargement d'images aux formats JPEG, PNG, BMP, TIFF et DICOM (extensions `.dcm` et `.dicom`). La taille maximale de fichier est fixée à 50 Mo. Le fichier est stocké temporairement en mémoire, transmis au pipeline d'IA, puis le fichier temporaire est supprimé après traitement.

#### Module d'anonymisation par IA

Le cœur fonctionnel est le pipeline d'anonymisation en sept étapes : (1) classification de l'image par CLIP, (2) validation du format, (3) anonymisation des métadonnées DICOM, (4) prétraitement CLAHE, (5) détection OCR double moteur, (6) suppression des régions textuelles, et (7) sauvegarde dans MinIO. Ce pipeline est paramétrable depuis l'interface utilisateur : seuil de confiance OCR (`conf_threshold`), marge de sécurité (`border_margin`), rembourrage (`padding`) et pourcentage de scan bordural (`border_pct`).

#### Module de stockage sécurisé

Les images anonymisées doivent être stockées dans MinIO, organisées par catégorie anatomique et par date. L'accès aux images doit se faire exclusivement par URL pré-signée à durée limitée (24 heures). Aucune image originale ne doit être persistée dans le système de stockage permanent.

#### Module de gestion et consultation

Chaque utilisateur doit pouvoir consulter l'historique de ses opérations avec les métadonnées associées (date, durée de traitement, nombre de régions détectées et supprimées, catégorie de l'image, lien de téléchargement). Les utilisateurs médicaux et administrateurs ont accès à l'historique global.

#### Module de reporting et traçabilité

Toutes les opérations doivent être journalisées en MongoDB avec : identifiant de l'utilisateur, date et heure, résultat (succès ou échec), paramètres utilisés, statistiques de traitement et identifiant MinIO.

### Besoins non fonctionnels

#### Performances

Le temps de traitement d'une image JPEG standard (500 Ko–2 Mo) ne doit pas excéder 5 secondes sur le matériel de développement (CPU, sans GPU). Pour les fichiers DICOM volumeux, un délai pouvant atteindre 15 secondes est acceptable. L'API doit pouvoir traiter au moins 5 requêtes concurrentes sans dégradation significative.

#### Sécurité

Toutes les routes API, à l'exception de l'authentification et de l'enregistrement, doivent être protégées par JWT. La gestion des rôles doit être implémentée côté serveur. Les mots de passe doivent être hachés avec bcrypt (facteur de coût 12). Aucune donnée sensible ne doit apparaître dans les journaux applicatifs.

#### Scalabilité et déploiement

L'architecture doit permettre une mise à l'échelle horizontale par ajout de nœuds de traitement, sans modification de la logique applicative. L'utilisation de Docker et Docker Compose doit faciliter le déploiement sur différentes infrastructures.

#### Conformité réglementaire

Le système doit assurer la conformité aux exigences du RGPD : minimisation des données (pas de stockage d'originaux), droit à l'effacement (suppression par l'administrateur), traçabilité des accès et protection des données dès la conception.

## Spécification fonctionnelle

### Cas d'utilisation principaux

Les acteurs du système sont : l'**Utilisateur** (médecin, technicien), l'**Utilisateur médical** (accès étendu), le **Responsable** (administrateur) et le **Système IA** (pipeline automatisé).

![Diagramme de cas d'utilisation global](example-image)

#### CU01 : S'authentifier

**Acteur principal :** Tout utilisateur enregistré.  
**Précondition :** L'utilisateur possède un compte valide dans le système.  
**Scénario nominal :** L'utilisateur accède à la page de connexion et saisit son adresse email et son mot de passe. Le système vérifie les identifiants en comparant le hash bcrypt du mot de passe fourni au hash stocké en base de données. Si les identifiants sont valides, le système génère un token JWT signé avec la clé secrète HMAC-SHA256 et l'envoie en réponse. Le client stocke le token et l'inclut dans l'en-tête `Authorization: Bearer` de chaque requête ultérieure.  
**Scénarios alternatifs :** Identifiants incorrects (HTTP 401) ; compte inexistant (HTTP 401) ; token expiré (HTTP 401, redirection vers la page de connexion).  
**Postcondition :** L'utilisateur est authentifié et peut accéder aux fonctionnalités correspondant à son rôle.

#### CU02 : Anonymiser une image

**Acteur principal :** Utilisateur authentifié.  
**Précondition :** L'utilisateur est connecté ; le fichier est dans un format supporté ; la taille est inférieure à 50 Mo.  
**Scénario nominal :** L'utilisateur sélectionne une image depuis son poste de travail et ajuste optionnellement les paramètres du pipeline via les curseurs de l'interface. Il soumet la requête. Le backend Node.js valide le fichier et les paramètres via Multer, puis transmet l'ensemble au pipeline FastAPI sous forme de `multipart/form-data`. Le pipeline exécute les sept étapes et retourne un JSON avec les statistiques et l'URL pré-signée MinIO.  
**Postcondition :** L'image anonymisée est stockée dans MinIO et l'opération est journalisée en MongoDB.

#### CU03 : Consulter l'historique

**Acteur principal :** Tout utilisateur authentifié.  
**Scénario nominal :** L'utilisateur accède au tableau de bord et consulte la liste de ses opérations, ordonnées par date décroissante, avec statistiques et lien de téléchargement.

#### CU04 : Administrer le système

**Acteur principal :** Responsable (administrateur).  
**Scénario nominal :** Le responsable consulte les statistiques globales, gère les comptes utilisateurs (modification du rôle, suppression), consulte les journaux d'audit et peut télécharger toutes les images anonymisées.

### Diagramme de séquence : Processus d'anonymisation

![Diagramme de séquence du processus d'anonymisation complet](example-image)

Le diagramme ci-dessus illustre le flux complet depuis la soumission de l'image par l'utilisateur jusqu'à la réception de l'URL de téléchargement. L'interaction entre les trois couches du système (frontend React, backend Node.js, pipeline FastAPI) y est clairement représentée, ainsi que les appels vers MinIO et MongoDB.

## Spécification technique

### Architecture générale

L'architecture du système repose sur une organisation en trois niveaux distincts et communicants : couche présentation (React), couche applicative (Node.js/Express), et couche traitement IA (FastAPI/Python). Cette séparation garantit l'indépendance technologique, la sécurité en profondeur et la modularité.

### Choix technologiques justifiés

#### Stack MERN pour l'application web

La pile MERN (MongoDB, Express, React, Node.js) a été choisie pour sa cohérence technologique (JavaScript/JSON de bout en bout), la richesse de son écosystème npm, sa forte adoption dans l'industrie et sa compatibilité avec les compétences de l'équipe. MongoDB, en tant que base de données NoSQL orientée documents, offre la flexibilité nécessaire pour stocker des journaux d'audit aux structures évolutives.

#### FastAPI pour le pipeline IA

FastAPI a été retenu comme framework Python pour plusieurs raisons déterminantes : performances quasi équivalentes à Node.js grâce à son architecture asynchrone (ASGI/Starlette), validation automatique des entrées via Pydantic, génération automatique de documentation OpenAPI/Swagger, et compatibilité native avec les bibliothèques de deep learning Python (PyTorch, OpenCV, PaddleOCR).

#### PaddleOCR et EasyOCR

PaddleOCR PP-OCRv4 (Baidu) et EasyOCR ont été retenus comme moteurs OCR primaire et secondaire respectivement. PaddleOCR excelle dans la détection de texte structuré et dense. EasyOCR, spécialisé sur les régions bordurales, complète efficacement les cas non détectés par le premier moteur. La fusion par IoU (seuil 0,5) élimine les doublons sans perte de couverture.

#### CLIP ViT-B/32 pour la classification

Le modèle CLIP ViT-B/32 d'OpenAI, accessible via HuggingFace Transformers, est utilisé pour la classification zéro-shot des images en cinq catégories anatomiques plus une catégorie _non_medical_, permettant le rejet automatique des images inappropriées.

#### MinIO pour le stockage objet

MinIO offre une compatibilité totale avec l'API Amazon S3, permettant un déploiement on-premises conforme aux exigences de souveraineté des données médicales. La génération d'URLs pré-signées limite l'exposition des images à des fenêtres temporelles contrôlées.

## Planification du projet

### Découpage en phases

Le développement a été organisé en cinq phases : (1) analyse et étude de l'art (2 semaines), (2) mise en place de l'architecture (1 semaine), (3) développement du pipeline IA (3 semaines), (4) développement de l'application web (3 semaines), et (5) tests, intégration et documentation (2 semaines).

![Diagramme de Gantt du projet](example-image)

### Gestion des risques

| Risque | Probabilité | Impact | Mesure d'atténuation |
|--------|-------------|--------|----------------------|
| Incompatibilité des bibliothèques Python | Élevée | Élevé | Environnement virtuel avec versions fixées dans `requirements.txt` |
| Performances insuffisantes de l'OCR | Moyenne | Élevé | Double moteur OCR et paramètres configurables depuis l'UI |
| Non-conformité RGPD | Faible | Très élevé | Audit de sécurité régulier, suppression des fichiers originaux |
| Dépendance au cloud externe | Faible | Moyen | Architecture déployable sur site avec MinIO |
| Échec de classification CLIP | Moyenne | Moyen | Fallback manuel et journalisation des rejets |

## Conclusion

Ce chapitre a formalisé les besoins fonctionnels et non fonctionnels du système, présenté les cas d'utilisation principaux et justifié les choix technologiques. Le chapitre suivant présente la conception détaillée du système, de la base de données et des modules d'intelligence artificielle.

---

_(Le document continue avec les chapitres 3-6, conclusion et annexes dans le même format Markdown...)_

**Note:** Le document complet fait plus de 15 000 lignes. Voulez-vous que je continue avec les chapitres restants (3-6, conclusion, annexes) ou préférez-vous que je crée un fichier séparé pour chaque chapitre ?
