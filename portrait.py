import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndi

# --- 1. Création des images binaires ---
def create_binary_image(data, filename):
    """
    Crée une image binaire et l'enregistre avec le nom spécifié.
    """
    img = Image.fromarray(np.array(data, dtype=np.uint8) * 255)  # Conversion pour affichage correct
    img.save(filename)

# Matrices d'exemple pour I1 et I2
I1 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

I2 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Créer et enregistrer les images binaires
create_binary_image(I1, 'I1.bmp')
create_binary_image(I2, 'I2.bmp')


# --- 2. Opérations d'addition et de soustraction ---
I_ad = I1 + I2  # Addition des matrices
I_s = I1 - I2  # Soustraction des matrices

# Créer et enregistrer les résultats des opérations
create_binary_image(I_ad, 'I_ad.bmp')
create_binary_image(I_s, 'I_s.bmp')


# --- 3. Affichage des résultats ---
def show_result(image_data, title):
    """
    Affiche une matrice binaire sous forme d'image.
    """
    plt.imshow(image_data, cmap='gray')
    plt.title(title)
    plt.axis('off')  # Désactiver les axes pour une meilleure vue
    plt.show()

# Afficher les matrices I1 et I2
show_result(I1, 'Image I1')
show_result(I2, 'Image I2')

# Afficher les résultats des opérations I_ad et I_s
show_result(I_ad, 'Résultat de I1 + I2 (Addition)')
show_result(I_s, 'Résultat de I1 - I2 (Soustraction)')


# --- 4. Fonction d'affichage de l'histogramme pour une image couleur ---
def display_histogram(image_path):
    """
    Affiche l'histogramme d'une image couleur (RGB).
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir l'image de BGR à RGB
    colors = ['r', 'g', 'b']  # Liste des couleurs (rouge, vert, bleu)

    # Calcul et affichage de l'histogramme pour chaque couleur
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])  # Calcul de l'histogramme
        plt.plot(histogram, color=color)

    plt.title('Histogramme de l\'image couleur')
    plt.xlabel('Intensité')
    plt.ylabel('Nombre de pixels')
    plt.show()


# Exemple d'affichage de l'histogramme pour une image
display_histogram('me.jpg')  # Remplacez par le chemin de votre image


# --- 5. Conversion en niveaux de gris ou binaire ---
def convert_image(image_path, output_path, mode='gray', threshold=128, bit_depth=8):
    """
    Convertit une image en niveaux de gris ou en image binaire, et enregistre le résultat.
    """
    image = cv2.imread(image_path)
    
    if mode == 'gray':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
        # Appliquer un bit-depth spécifique si nécessaire
        gray_image = (gray_image // (256 // (2 ** bit_depth))) * (256 // (2 ** bit_depth))
        cv2.imwrite(output_path, gray_image)  # Enregistrer l'image
    elif mode == 'binary':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)  # Seuil binaire
        cv2.imwrite(output_path, binary_image)  # Enregistrer l'image binaire


# Exemple de conversion en niveaux de gris et en binaire
convert_image('me.jpg', 'portrait_gray.jpg', mode='gray', bit_depth=4)
convert_image('me.jpg', 'portrait_binary.jpg', mode='binary', threshold=100)


# --- 6. Application du filtre de Nagao ---
def apply_nagao_filter(image_path, output_path):
    """
    Applique un filtre de Nagao à une image et enregistre le résultat.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Lire l'image en niveaux de gris
    # Appliquer le filtre de Nagao (utilisation de la variance locale comme filtre)
    nagao_image = ndi.generic_filter(image, np.var, size=5)  # Taille du filtre : 5x5
    cv2.imwrite(output_path, nagao_image)  # Enregistrer l'image filtrée


# Appliquer le filtre de Nagao à l'image en niveaux de gris
apply_nagao_filter('portrait_gray.jpg', 'portrait_nagao.jpg')
