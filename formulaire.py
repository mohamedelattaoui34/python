import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter


# --- Question 1 : Création et manipulation d'images binaires ---
def create_binary_images():
    # Création des deux images binaires I1 et I2
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

    # Addition spécifique
    I_ad = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    I_s = np.clip(I1 - I2, 0, 1)

    # Affichage des résultats
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Image I1")
    plt.imshow(I1, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Image I2")
    plt.imshow(I2, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Addition (I_ad)")
    plt.imshow(I_ad, cmap='gray')
    plt.show()

    return I1, I2, I_ad, I_s


def multiply_image_gray():
    # Image de gris donnée
    I_gray = np.array([
        [0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0],
        [0 ,3 ,3 ,4 ,4 ,5 ,5 ,6 ,0],
        [0 ,3 ,3 ,4 ,4 ,5 ,5 ,6 ,0],
        [0 ,6 ,6 ,5 ,5 ,4 ,4 ,3 ,0],
        [0 ,7 ,8 ,9 ,7 ,8 ,9 ,7 ,0],
        [0 ,9 ,9 ,8 ,8 ,7 ,7 ,7 ,0],
        [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],
        [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
    ])
    # Multiplication par 2
    I_p = np.clip(I_gray * 2, 0, 15)

    # Affichage
    plt.imshow(I_p, cmap='gray')
    plt.title("Image Multipliée par 2")
    plt.colorbar()
    plt.show()

    return I_p


# --- Question 2 : Histogramme d'une image couleur ---
def display_histogram(image_path):
    # Charger une image couleur
    image = Image.open(image_path)
    image_np = np.array(image)

    # Calcul et affichage de l'histogramme
    plt.figure(figsize=(10, 5))
    plt.title("Histogramme de l'image couleur")
    for i, color in enumerate(['r', 'g', 'b']):
        histo, _ = np.histogram(image_np[..., i], bins=256, range=(0, 255))
        plt.plot(histo, color=color, label=f"Canal {color.upper()}")
    plt.legend()
    plt.show()


# --- Question 3 : Transformation d'une image ---
def transform_image(image_path, mode='gray', bits=8):
    img = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
    img_np = np.array(img)

    if mode == 'binary':
        threshold = 127  # Seuil pour binaire
        img_binary = (img_np > threshold).astype(np.uint8) * 255
        return Image.fromarray(img_binary)
    elif mode == 'gray':
        factor = 2 ** (8 - bits)
        img_reduced = (img_np // factor) * factor
        return Image.fromarray(img_reduced)


# --- Question 4 : Histogramme de l'image transformée ---
def display_transformed_histogram(transformed_image):
    transformed_image_np = np.array(transformed_image)
    plt.hist(transformed_image_np.ravel(), bins=256, range=(0, 255))
    plt.title("Histogramme de l'image transformée")
    plt.show()


# --- Question 5 : Filtre de Nagao ---
def nagao_filter(image):
    """Appliquer le filtre de Nagao sur une image en niveaux de gris."""
    image = np.array(image, dtype=np.float32)
    regions = [
        uniform_filter(image, size=3, mode='reflect'),
        uniform_filter(image[:-2, :-2], size=3, mode='reflect'),
        uniform_filter(image[:-2, 1:-1], size=3, mode='reflect'),
        uniform_filter(image[1:-1, :-2], size=3, mode='reflect'),
        uniform_filter(image[1:-1, 1:-1], size=3, mode='reflect'),
    ]
    filtered = np.min(regions, axis=0)
    return filtered


# --- Main : Appeler les fonctions ---
if __name__ == "__main__":
    # Question 1
    print("Question 1: Création et manipulation d'images binaires")
    I1, I2, I_ad, I_s = create_binary_images()
    multiply_image_gray()

    # Question 2
    print("Question 2: Histogramme d'une image couleur")
    display_histogram('me.jpg')  # Remplacez par le chemin de votre image

    # Question 3
    print("Question 3: Transformation d'une image")
    transformed_image = transform_image('me.jpg', mode='binary')
    transformed_image.show()

    # Question 4
    print("Question 4: Histogramme de l'image transformée")
    display_transformed_histogram(transformed_image)

    # Question 5
    print("Question 5: Application du filtre de Nagao")
    image_gray = Image.open('me.jpg').convert('L')
    nagao_result = nagao_filter(image_gray)
    plt.imshow(nagao_result, cmap='gray')
    plt.title("Image filtrée avec Nagao")
    plt.show()
