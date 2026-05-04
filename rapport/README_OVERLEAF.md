# Rapport PFE LaTeX — Instructions Overleaf

**Auteur :** Mouheb Boubrik  
**Établissement :** ISITCOM Hammam Sousse  
**Année :** 2025–2026

---

## 📁 Structure du projet

```
rapport/
├── main.tex                    ← Fichier principal (compiler celui-ci)
├── references.bib              ← Bibliographie BibTeX
├── README_OVERLEAF.md          ← Ce fichier
└── chapters/
    ├── titlepage.tex           ← Page de garde
    ├── dedicaces.tex           ← Dédicaces
    ├── remerciements.tex       ← Remerciements
    ├── resume.tex              ← Résumé FR + EN + AR
    ├── abbreviations.tex       ← Liste des abréviations
    ├── intro.tex               ← Introduction générale
    ├── chap1.tex               ← Chapitre 1 : Contexte & état de l'art
    ├── chap2.tex               ← Chapitre 2 : Analyse & spécification
    ├── chap3.tex               ← Chapitre 3 : Conception
    ├── chap4.tex               ← Chapitre 4 : Implémentation
    ├── chap5.tex               ← Chapitre 5 : Tests & validation
    ├── chap6.tex               ← Chapitre 6 : Résultats & discussion
    ├── conclusion.tex          ← Conclusion générale & perspectives
    └── annexes.tex             ← Annexes A, B, C, D
```

---

## 🚀 Comment utiliser sur Overleaf

### Étape 1 — Créer un nouveau projet

1. Aller sur [overleaf.com](https://www.overleaf.com)
2. Cliquer **New Project → Upload Project**
3. Zipper le dossier `rapport/` entier et uploader le zip

### Étape 2 — Configurer le compilateur

1. Dans Overleaf, cliquer sur **Menu** (en haut à gauche)
2. Sous **Compiler**, sélectionner **pdfLaTeX**
3. Sous **Main document**, sélectionner **main.tex**

### Étape 3 — Compiler

Cliquer sur **Recompile** — le document se compile en PDF sans aucune modification.

---

## 🖼️ Ajouter vos figures

Toutes les figures utilisent actuellement `example-image` (image de substitution grise).  
Pour remplacer avec vos vraies captures d'écran :

1. Uploader vos images dans Overleaf (formats PNG, JPG, PDF)
2. Remplacer `example-image` par le nom de votre fichier dans chaque `\includegraphics`

**Exemple :**
```latex
% Avant (placeholder)
\includegraphics[width=\textwidth]{example-image}

% Après (avec votre image)
\includegraphics[width=\textwidth]{dashboard_screenshot}
```

### Figures recommandées à préparer

| Figure | Contenu | Chapitre |
|--------|---------|----------|
| `fig_architecture` | Architecture globale 3 couches | Chap. 3 |
| `fig_pipeline` | Pipeline 7 étapes (diagramme) | Chap. 3 |
| `fig_clip` | Architecture CLIP zéro-shot | Chap. 3 |
| `fig_login` | Capture page connexion | Chap. 6 |
| `fig_dashboard` | Capture tableau de bord | Chap. 6 |
| `fig_results` | Carte de résultats | Chap. 6 |
| `fig_history` | Section historique | Chap. 6 |
| `fig_admin` | Panneau admin | Chap. 6 |
| `fig_avant_apres_1` | Image avant anonymisation | Chap. 6 |
| `fig_avant_apres_2` | Image après anonymisation | Chap. 6 |
| `fig_swagger` | Documentation Swagger | Annexe B |
| `fig_gantt` | Diagramme de Gantt | Chap. 2 |

---

## ✏️ Personnalisations à faire

### 1. Page de garde (`chapters/titlepage.tex`)
- Remplacer `example-image` par le vrai logo ISITCOM

### 2. Résumé arabe (`chapters/resume.tex`)
- La section arabe contient un placeholder — à compléter avec un traducteur ou en ajoutant le package `polyglossia` dans `main.tex`

### 3. Informations personnelles
- Vérifier que votre nom, les noms des encadreurs et l'année académique sont corrects dans `titlepage.tex`

---

## 📋 Checklist avant soumission

- [ ] Logo ISITCOM inséré sur la page de garde
- [ ] Toutes les figures `example-image` remplacées par les vraies captures
- [ ] Résumé en arabe complété
- [ ] Table des matières générée automatiquement (se fait à la compilation)
- [ ] Numéros de pages des figures et tableaux vérifiés
- [ ] Bibliographie vérifiée (toutes les références `\cite{}` présentes dans `references.bib`)
- [ ] Diagramme de Gantt inséré (Chapitre 2)
- [ ] Diagrammes UML insérés (use case, séquence, déploiement)

---

## 🔧 Packages requis

Tous les packages sont standards et disponibles sur Overleaf sans installation supplémentaire :

- `inputenc`, `fontenc`, `babel` — encodage et langue française
- `geometry` — marges 2,5 cm
- `times`, `microtype` — typographie professionnelle
- `graphicx`, `xcolor`, `float` — figures et couleurs
- `booktabs`, `longtable`, `tabularx` — tableaux
- `amsmath`, `amssymb` — mathématiques
- `listings` — blocs de code
- `tcolorbox` — encadrés colorés (page de garde)
- `fancyhdr` — en-têtes et pieds de page
- `titlesec` — formatage des titres
- `hyperref` — hyperliens et métadonnées PDF
- `caption`, `enumitem` — légendes et listes

---

## 📞 Contact

Pour toute question sur ce document LaTeX, contacter **Mouheb Boubrik**.
