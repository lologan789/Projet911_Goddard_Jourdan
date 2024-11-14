# Projet911_Goddard_Jourdan

## Projet de Reconnaissance d'Objets Basée sur les Histogrammes de Couleurs

Ce projet utilise OpenCV pour effectuer une reconnaissance d'objets dans une vidéo ou une image capturée en temps réel, en utilisant des histogrammes de couleurs pour identifier et classer des objets dans différentes régions de l'image. L'algorithme divise l'image en blocs et compare les histogrammes de couleurs avec ceux d'objets connus afin de classer chaque bloc.

## Fonctionnalités principales

**Capture vidéo en temps réel :** Utilisation de la caméra pour capturer des images en temps réel.

**Histogrammes de couleurs :** Calcul des histogrammes de couleurs pour les blocs d'image et les objets.

**Classification d'objets :** Classement des blocs d'image en fonction de la similarité de leurs histogrammes de couleurs avec des objets pré-définis.

**Ajout dynamique d'objets :** Possibilité d'ajouter de nouveaux objets avec leurs propres histogrammes et couleurs.

## Utilisation

**Commandes** 

q ou Esc : Quitter l'application.

f : Geler/dégeler l'image actuelle pour une analyse statique.

v : Calculer et afficher la distance entre les histogrammes de couleurs de la partie gauche et droite de l'image.

b : Calculer les histogrammes de couleurs pour l'arrière-plan.

a : Calculer et ajouter un histogramme pour une zone d'objet spécifique. (Ne fonctionne plus dans la derniere version)

n : Ajouter un nouvel objet avec son histogramme et une couleur aléatoire.

r : Activer le mode de reconnaissance.

## Mode de fonctionnement

Appuyer dans un premier temps sur b pour calculer les histogrammes de couleurs de l'arriere-plan puis positionner l'objet de votre choix devant le rectangle blanc, appuyer sur n pour l'ajouter a la liste des objets et lui assigner une couleur. Appuyer enfin sur r pour activer le mode de reconnaissance et visualiser votre objet sur le champs de votre camera.