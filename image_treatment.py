import cv2
from utils import get_resource_path
from tkinter import filedialog # Import the filedialog module to open file dialogs
import os
import numpy as np # SOLO PARA CONVERTIR LA IMAGEN EN UN FORMATO QUE CUSTOMTKINTER PUEDA MOSTRAR EN UN LABEL

def load_images(is_folder):
    """
    Load images from a folder or a single image.
    
    Parameters:
    is_folder (bool): True if the user wants to load a folder, False if the user wants to load a single image.
    
    Returns:
    images (list): List with the images loaded.
    """
    images = []
    try:
        if is_folder:
            folder = filedialog.askdirectory()
            folder = get_resource_path(folder)
            if folder:
                for filename in os.listdir(folder):
                    img_path = os.path.join(folder, filename)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                    if img is not None:
                        images.append(img)
        else:
            img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg")])
            if img_path:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                if img is not None:
                    images.append(img)
        return images
    except Exception as e:
        print("Error loading images: ", e)
        return images

def apply_kernel(image, kernel, padding=0, stride=1):
    """
    Aplica un kernel 3x3 o 5x5 sobre una imagen con padding y stride definidos.
    
    Args:
        image: La imagen en formato BGR.
        kernel: El kernel (matriz) 3x3 o 5x5 o 7x7.
        padding: Cantidad de padding a añadir a la imagen.
        stride: El número de píxeles que se avanza después de aplicar el kernel.
        
    Returns:
        La imagen convolucionada en formato compatible con Image.fromarray().
    """
    # Dimensiones de la imagen y del kernel
    image_height, image_width, _ = image.shape
    kernel_size = len(kernel)  # Asumimos kernel cuadrado (3x3 o 5x5)
    kernel_center = kernel_size // 2

    # Aplicar padding a la imagen
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Dimensiones de la imagen después de aplicar padding
    padded_height, padded_width, _ = padded_image.shape

    # Dimensiones de la salida
    output_height = (padded_height - kernel_size) // stride + 1
    output_width = (padded_width - kernel_size) // stride + 1

    # Imagen de salida (combinación de los tres canales BGR)
    output_image = [[[0, 0, 0] for _ in range(output_width)] for _ in range(output_height)]

    # Aplicar el kernel sobre cada canal de la imagen (BGR)
    for y in range(0, output_height):
        for x in range(0, output_width):
            # Inicializar acumuladores para cada canal
            new_pixel_b = 0.0
            new_pixel_g = 0.0
            new_pixel_r = 0.0

            # Aplicar el kernel en el área de la imagen
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    # Coordenadas en la imagen original con stride
                    pixel_y = y * stride + ky - kernel_center
                    pixel_x = x * stride + kx - kernel_center

                    if 0 <= pixel_y < padded_height and 0 <= pixel_x < padded_width:
                        # Obtener el valor de los píxeles BGR
                        pixel_b, pixel_g, pixel_r = padded_image[pixel_y, pixel_x].astype(float)

                        # Multiplicar por el kernel y sumar a los canales correspondientes
                        new_pixel_b += pixel_b * kernel[ky][kx]
                        new_pixel_g += pixel_g * kernel[ky][kx]
                        new_pixel_r += pixel_r * kernel[ky][kx]

            # Clipping de los valores para mantenerlos en el rango [0, 255]
            output_image[y][x] = [
                min(max(int(new_pixel_b), 0), 255),
                min(max(int(new_pixel_g), 0), 255),
                min(max(int(new_pixel_r), 0), 255)
            ]

    # Convertir la lista 'output_image' en un array de OpenCV (que Image.fromarray pueda usar)
    output_image_array = np.array(output_image, dtype=np.uint8)  # Convertimos a un array compatible
    print("Imagen correctamente convulsionada")

    return output_image_array  # Devolver la imagen convertida en formato array

def get_kernel(name = 'rgb_to_bgr', p = 0):
    
    kernels = {
        "3x3_Identity": [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
        "3x3_Gaussian Blur": [
            [1/16, 2/16, 1/16],
            [2/16, 4/16, 2/16],
            [1/16, 2/16, 1/16]
        ],
        "5x5_Gaussian Blur": [
            [1/256,  4/256,  6/256,  4/256,  1/256],
            [4/256, 16/256, 24/256, 16/256,  4/256],
            [6/256, 24/256, 36/256, 24/256,  6/256],
            [4/256, 16/256, 24/256, 16/256,  4/256],
            [1/256,  4/256,  6/256,  4/256,  1/256]
        ],
        "7x7_gaussian_blur": [
            [0/1003, 0/1003, 1/1003, 2/1003, 1/1003, 0/1003, 0/1003],
            [0/1003, 3/1003, 13/1003, 22/1003, 13/1003, 3/1003, 0/1003],
            [1/1003, 13/1003, 59/1003, 97/1003, 59/1003, 13/1003, 1/1003],
            [2/1003, 22/1003, 97/1003, 159/1003, 97/1003, 22/1003, 2/1003],
            [1/1003, 13/1003, 59/1003, 97/1003, 59/1003, 13/1003, 1/1003],
            [0/1003, 3/1003, 13/1003, 22/1003, 13/1003, 3/1003, 0/1003],
            [0/1003, 0/1003, 1/1003, 2/1003, 1/1003, 0/1003, 0/1003]
        ],
        "3x3_Sobel Vertical": [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        "5x5_Sobel Vertical": [
            [-1, -2, 0, 2, 1],
            [-4, -8, 0, 8, 4],
            [-6, -12, 0, 12, 6],
            [-4, -8, 0, 8, 4],
            [-1, -2, 0, 2, 1]
        ],
        "3x3_Sobel Horizontal": [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        "5x5_Sobel Horizontal": [
            [-1, -4, -6, -4, -1],
            [-2, -8, -12, -8, -2],
            [0, 0, 0, 0, 0],
            [2, 8, 12, 8, 2],
            [1, 4, 6, 4, 1]
        ],
        "3x3_Laplacian": [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ],
        "5x5_Laplacian": [
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ],
        "3x3_Prewitt": [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        "5x5_Prewitt": [
            [-1, -1, 0, 1, 1],
            [-2, -2, 0, 2, 2],
            [-3, -3, 0, 3, 3],
            [-2, -2, 0, 2, 2],
            [-1, -1, 0, 1, 1]
        ],
        "3x3_Sharpen": [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        "5x5_Sharpen": [
            [0, -1, -1, -1, 0],
            [-1, 2, -2, 2, -1],
            [-1, -2, 12, -2, -1],
            [-1, 2, -2, 2, -1],
            [0, -1, -1, -1, 0]
        ],
        "3x3_Emboss": [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ],
        "5x5_Emboss": [
            [-4, -2, 0, 2, 4],
            [-2, -1, 0, 1, 2],
            [0, 0, 0, 0, 0],
            [2, 1, 0, -1, -2],
            [4, 2, 0, -2, -4]
        ],
        "3x3_Box Blur": [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ],
        "5x5_Box Blur": [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ],
        "3x3_High-Pass Filter": [
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ],
        "5x5_High-Pass Filter": [
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 24, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1]
        ],
        "3x3_Motion Blur": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ],
        "5x5_Motion Blur": [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ],
        "3x3_Edge Detection (Roberts Cross)": [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 0]
        ],
        "5x5_Edge Detection (Roberts Cross)": [
            [1, 1, 0, -1, -1],
            [1, 1, 0, -1, -1],
            [0, 0, 0, 0, 0],
            [-1, -1, 0, 1, 1],
            [-1, -1, 0, 1, 1]
        ],
        "3x3_Diagonal Edge Detection": [
            [2, -1, -1],
            [-1, 2, -1],
            [-1, -1, 2]
        ],
        "5x5_Diagonal Edge Detection": [
            [3, 3, 0, -3, -3],
            [3, 3, 0, -3, -3],
            [0, 0, 0, 0, 0],
            [-3, -3, 0, 3, 3],
            [-3, -3, 0, 3, 3]
        ],
        "3x3_Laplacian Sharpen": [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        "5x5_Laplacian Sharpen": [
            [0, -1, -1, -1, 0],
            [-1, 2, -2, 2, -1],
            [-1, -2, 12, -2, -1],
            [-1, 2, -2, 2, -1],
            [0, -1, -1, -1, 0]
        ],
        "3x3_Gabor Filter": [
            [1, 0, -1],
            [0, 0, 0],
            [-1, 0, 1]
        ],
        "5x5_Gabor Filter": [
            [1, 1, 0, -1, -1],
            [1, 2, 0, -2, -1],
            [0, 0, 0, 0, 0],
            [-1, -2, 0, 2, 1],
            [-1, -1, 0, 1, 1]
        ],
        "7x7_Gabor Filter": [
            [1, 1, 1, 0, -1, -1, -1],
            [1, 2, 2, 0, -2, -2, -1],
            [1, 2, 3, 0, -3, -2, -1],
            [0, 0, 0, 0, 0, 0, 0],
            [-1, -2, -3, 0, 3, 2, 1],
            [-1, -2, -2, 0, 2, 2, 1],
            [-1, -1, -1, 0, 1, 1, 1]
        ],
        "3x3_Edge Detection (Laplacian of Gaussian)": [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ],
        "3x3_Edge Detection (Laplacian of Gaussian) plus": [
            [0, 2, 0],
            [2, -8, 2],
            [0, 2, 0]
        ],
        "3x3_Edge Detection (Laplacian of Gaussian) plus2": [
            [0, 1.5, 0],
            [1.5, -6, 1.5],
            [0, 1.5, 0]
        ],
        "5x5_Edge Detection (Laplacian of Gaussian)": [
            [0,  0, -1,  0,  0],
            [0, -1, -2, -1,  0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1,  0],
            [0,  0, -1,  0,  0]
        ],
        "3x3_Average Blur": [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ],
        "5x5_Average Blur": [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ],
        "3x3_Edge Enhancement": [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        "3x3_Edge Enhancement less": [
            [0, -0.5, 0],
            [-0.5, 2.5, -0.5],
            [0, -0.5, 0]
        ],
        "5x5_Edge Enhancement": [
            [0, -1, -1, -1,  0],
            [-1,  2, -2,  2, -1],
            [-1, -2, 12, -2, -1],
            [-1,  2, -2,  2, -1],
            [0, -1, -1, -1,  0]
        ],
        "3x3_Outline Filter": [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ],
        "5x5_Outline Filter": [
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 24, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1]
        ],
        "3x3_Ridge Detection": [
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ],
        "5x5_Ridge Detection": [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, -24, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ],
        "3x3_Gaussian Sharpen": [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        "5x5_Gaussian Sharpen": [
            [0, -1, -1, -1, 0],
            [-1,  2, -2,  2, -1],
            [-1, -2, 12, -2, -1],
            [-1,  2, -2,  2, -1],
            [0, -1, -1, -1, 0]
        ],
        "3x3_High Boost Filter": [
            [1, -1, 1],
            [-1, 5, -1],
            [1, -1, 1]
        ],
        "5x5_High Boost Filter": [
            [1, -1, -1, -1, 1],
            [-1, 2, -2, 2, -1],
            [-1, -2, 12, -2, -1],
            [-1, 2, -2, 2, -1],
            [1, -1, -1, -1, 1]
        ],
        "3x3_Edge Detection (Kirsch)": [
            [-3, -3, 5],
            [-3,  0, 5],
            [-3, -3, 5]
        ],
        "5x5_Edge Detection (Kirsch)": [
            [-3, -3, 0, 3, 3],
            [-3, -3, 0, 3, 3],
            [0,  0,  0,  0, 0],
            [3,  3, 0, -3, -3],
            [3,  3, 0, -3, -3]
        ],
        "3x3_Edge Detection (Robinson)": [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ],
        "5x5_Edge Detection (Robinson)": [
            [-1, -2,  0,  2,  1],
            [-2, -4,  0,  4,  2],
            [0,  0,  0,  0,  0],
            [2,  4,  0, -4, -2],
            [1,  2,  0, -2, -1]
        ],
        "3x3_less_white": [
            [p/256, p/256, p/256],
            [p/256, p/256, p/256],
            [p/256, p/256, p/256]
        ],
        "less_blue": reduce_blue,
        "less_green":reduce_green,
        "less_red": reduce_red,
        "more_blue": increase_blue,
        "more_green": increase_green,
        "more_red": increase_red,
        "rgb_to_bgr": rgb_to_bgr,
        "bgr_to_rgb": bgr_to_rgb,
        "gray_to_rgb": gray_to_rgb,
        "gray_to_bgr": gray_to_bgr,
        "rgb_to_gray": rgb_to_gray,
        "bgr_to_gray": bgr_to_gray,
    }
    
    return kernels[name]

def rgb_to_bgr(image):
    return image[:, :, ::-1]

def bgr_to_rgb(image):
    return image[:, :, ::-1]

def gray_to_rgb(image):
    return np.stack((image, image, image), axis=-1)

def gray_to_bgr(image):
    return np.stack((image, image, image), axis=-1)

def rgb_to_gray(image):
    # Dimensiones de la imagen
    height, width, _ = image.shape
    
    # Crear una nueva imagen con 3 canales (R=G=B)
    gray_image = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]
    
    # Convertir cada píxel a escala de grises y replicar el valor en los tres canales
    for y in range(height):
        for x in range(width):
            r, g, b = image[y][x].astype(float)
            # Calcular el valor de gris usando la fórmula
            gray_value = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
            gray_value = min(255, max(0, gray_value))  # Clipping para mantener el valor en el rango [0, 255]w
            # Asignar el mismo valor a los tres canales
            gray_image[y][x] = [gray_value, gray_value, gray_value]
    
    return np.array(gray_image, dtype=np.uint8)

def bgr_to_gray(image):
    gray_image = (0.1140 * image[:, :, 0]) + (0.5870 * image[:, :, 1]) + (0.2989 * image[:, :, 2])
    return gray_image.astype(np.uint8)

def reduce_red(image, p=0.5):
    image[:, :, 0] = (image[:, :, 0] * p).astype('uint8')
    return image

def reduce_green(image, p=0.5):
    image[:, :, 1] = (image[:, :, 1] * p).astype('uint8')
    return image

def reduce_blue(image, p=0.5):
    image[:, :, 2] = (image[:, :, 2] * p).astype('uint8')
    return image

def increase_red(image, intensity):
    output_image = []
    for row in image:
        new_row = []
        for pixel in row:
            r = min(255, int(pixel[0] * (1 + intensity)))
            new_row.append([r, pixel[1], pixel[2]])
        output_image.append(new_row)
    return np.array(output_image, dtype=np.uint8)

def increase_green(image, intensity):
    output_image = []
    for row in image:
        new_row = []
        for pixel in row:
            g = min(255, int(pixel[1] * (1 + intensity)))
            new_row.append([pixel[0], g, pixel[2]])
        output_image.append(new_row)
    return np.array(output_image, dtype=np.uint8)

def increase_blue(image, intensity):
    output_image = []
    for row in image:
        new_row = []
        for pixel in row:
            b = min(255, int(pixel[2] * (1 + intensity)))
            new_row.append([pixel[0], pixel[1], b])
        output_image.append(new_row)
    return np.array(output_image, dtype=np.uint8)

def download_images(tag_name, images):
    """
        Funtion to download images in a folder.
        all images will be saved in the same pgn format.
    """
    
    if not images or len(images) == 0:
        print("No images to download")
        return
    
    if not tag_name or tag_name == "":
        print("Tag name is required")
        return
    
    print("Downloading images...")
    # Print the shape of the image
    print("Resolución: ", images[0].shape)
    
    print("Matrix of one image: ", images[0])
    
    # Get the path to save the image
    folder = filedialog.askdirectory()
    folder = get_resource_path(folder)
    if folder:
        try:
            for i, img in enumerate(images):
                # Save the image
                cv2.imwrite(f"{folder}/{tag_name}_{i}.png", img)
        except Exception as e:
            print("Error saving the images", e) 
            
    print("Images downloaded successfully")
    
def plain_image(image):
    """
    Function to convert an image to a vector without numpy.
    
    Parameters:
    image (np.array): The image to convert.
        
    Returns:
    vector (list): The image as a vector.
    """
    
    print("Converting image to vector...")
    
    vector = []
    for row in image:
        for pixel in row:
            vector.extend(pixel)
    
    print("Image converted to vector")
    return vector

def normalize_image_vector(image_vector):
    """
    Normalize the image vector.
    Where the max value is 255.
    
    Parameters:
    image_vector (list): The image as a vector.
    
    Returns:
    image_vector (list): The image vector normalized.
    """
    
    for i in range(len(image_vector)):
        image_vector[i] = image_vector[i] / 255
        
    return image_vector

def resize_image(image, width, height, interpolation):
    """
    Resize an image.
    
    Parameters:
    image (np.array): The image to resize.
    width (int): The new width.
    height (int): The new height.
    
    Returns:
    image_resized (np.array): The resized image.
    """
    
    if image is None:
        print("No image to resize")
        return

    if width <= 0 or height <= 0:
        print("Invalid width or height")
        return
    
    if interpolation not in ["Nearest", "Bilinear", "Bicubic", "Area-based", "Lanczos", "Spline"]:
        print("Invalid interpolation method")
        return
    
    if width == image.shape[1] and height == image.shape[0]:
        print("The image is already in the desired size")
        return image
    
    print("Resizing image...", image.shape)
    
    if interpolation == "Nearest":
        interpolation = cv2.INTER_NEAREST
    elif interpolation == "Bilinear":
        interpolation = cv2.INTER_LINEAR
    elif interpolation == "Bicubic":
        interpolation = cv2.INTER_CUBIC
    elif interpolation == "Area-based":
        interpolation = cv2.INTER_AREA
    elif interpolation == "Lanczos":
        interpolation = cv2.INTER_LANCZOS4
    elif interpolation == "Spline":
        interpolation = cv2.INTER_CUBIC
    
    image_resized = cv2.resize(image, (width, height), interpolation = interpolation)
    
    print(f"Resized image shape: {image_resized.shape}")
    
    return image_resized