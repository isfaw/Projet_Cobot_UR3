# Projet - Prédiction des arrêts de protection d’un Cobot UR3

## Étudiant
Isfaw Ait Kerrou  
isfaw.aitkerrou@centrale-casablanca.ma

## Méthodologie
- Prétraitement des données (imputation, normalisation)
- Création de séquences temporelles (taille 10)
- Modèles entraînés : LSTM, Random Forest, SVM
- Rééchantillonnage SMOTE pour données déséquilibrées

## Comparaison des modèles
- LSTM : AUC = 0.512687, F1 = 0.333333
- Random Forest : AUC = 0.993823, F1 = 0.950581	
- SVM : AUC = 0.982388, F1 = 0.948513

## Lancer l’API Flask
```bash
docker build -t mon-api .
docker run -p 5000:5000 mon-api
