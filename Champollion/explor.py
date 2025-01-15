import pandas as pd

# Chemin vers le fichier CSV
csv_path1 = "E:/projects hetic/deeplearning/Champollion/test/_annotations.csv"
csv_path2 = "E:/projects hetic/deeplearning/Champollion/train/_annotations.csv"
csv_path3 = "E:/projects hetic/deeplearning/Champollion/valid/_annotations.csv"


# Lire le fichier CSV
annotations1 = pd.read_csv(csv_path1)
annotations2 = pd.read_csv(csv_path2)
annotations3 = pd.read_csv(csv_path3)
# Afficher la première ligne sous forme de tableau lisible
print(annotations1.head(1))  # head(1) affiche les premières lignes du DataFrame

# Compter le nombre d'images distinctes
unique_images1 = annotations1['filename'].nunique()
unique_images2 = annotations2['filename'].nunique()
unique_images3 = annotations3['filename'].nunique()
# Afficher le résultat
print(f"Nombre total d'images distinctes du test : {unique_images1}")
# Afficher le résultat
print(f"Nombre total d'images distinctes du train : {unique_images2}")
# Afficher le résultat
print(f"Nombre total d'images distinctes de la validaion : {unique_images3}")