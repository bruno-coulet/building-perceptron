# Comprendre les données médicales

Ce jeu de données contient des mesures obtenues à partir d'images microscopiques de noyaux cellulaires prélevés lors d'une biopsie mammaire.

L'objectif est de prédire si une tumeur est : 

	* Bégnigne (B) : non cancéreuse
	* Maligne (M) : cancéreuse

Chaque ligne représente une patiente et chaque colonne décrit une caractéristique morphologique observée sur les cellules.

# Variable cible

diagnosis (diagnostique médical réel) 
id (identifiant unique de l'observation)

# Les caractéristiques mesurées

Chaque caractéristique est calculée selin trois versions : 

	* mean : valeur moyenne observée
	* se (standar error) : variabilité de la mesure
	* worst : valeur la plus extême observée

# Significaiton des mesures

## Radius

radius_* : Distance moyenne entre le centre du noyeau cellulaire et son contour

Un rayon élevé indique générament des cellulles plus volumineuses.

## Texture

texture_* : Variabilité des niveau de gris dans l'image

Mesure l'hétérogénéité visuelle du noyau. 

# Perimeter

perimeter_* : Surface du noyau cellulaire

Les tumeurs malignes présentes souvent des surfaces plus importantes.

# Smoothness

smoothness_* : Régularité du contour

Une valeur élevée indique des variations plus rapides du contour. 

## Compactness

compactness_** : Comactité de la forme (périmètre^2 / aire)

Des valeurs élevée traduisent des formes plus irrégulières.

## Concavity

concavity_* : Importance des creus présents sur le contour

Une forte concavité est souvent associées à des cellules normales.

## Concave Points

concave points_* : Nombre et importance des points de concavité

Mesure les zeones où le contour s'enfonce ver l'intérieur. 

Cette variable est souvent parmis les plus discriminantes pour détecter les tumeurs malignes. 

## Symetry 

symetry_* : Symétrie globale du noyau

Les noyau sains sont généralement plus symétriques.

## Fractal Dimension 

fractal_dimension_* : Complexité géométriue du contour

Plus cette valeur est élevée, plus le contour présente des irrégularités fines. 

