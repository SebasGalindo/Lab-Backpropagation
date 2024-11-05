# Description: Image processing functions for the Image Processing GUI
# Authors: John Sebastián Galindo Hernández, Miguel Ángel Moreno Beltrán

# region Import libraries
import cv2
from utils import get_resource_path
from tkinter import filedialog # Import the filedialog module to open file dialogs
import os
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import correlate
# endregion

# region Image lecture and download
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
                    try:
                        img_path = os.path.join(folder, filename)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                        if img is not None:
                            print("Image with path: ", img_path, " loaded")
                            images.append(img)
                    except Exception as e:
                        print("Error loading image: ", e)
        else:
            img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.ppm;*.pgm;*.pbm;*.webp")])
            if img_path:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                if img is not None:
                    images.append(img)
        return images
    except Exception as e:
        print("Error loading images: ", e)
        return images

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
                bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{folder}/{tag_name}_{i}.png", bgr_image)
        except Exception as e:
            print("Error saving the images", e) 
            
    print("Images downloaded successfully")
# endregion

# region Normal kernel selection and application
def apply_kernel_normal(image, kernel, padding=0, stride=1):
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

def apply_kernel_numpy(image, kernel, padding=0, stride=1, index=0):
    """
    Aplica un kernel sobre una imagen con padding y stride definidos utilizando numpy.
    
    Args:
        image: La imagen en formato BGR.
        kernel: El kernel (matriz) de tamaño variable (3x3, 5x5, etc.).
        padding: Cantidad de padding a añadir a la imagen.
        stride: El número de píxeles que se avanza después de aplicar el kernel.
        
    Returns:
        La imagen convolucionada en formato compatible con Image.fromarray().
    """
    
    print(f"Aplicando kernel a la imagen {index}...")
    
    # Dimensiones de la imagen y del kernel
    image_height, image_width, _ = image.shape
    kernel_size = len(kernel)  # Asumimos kernel cuadrado (3x3, 5x5, etc.)
    
    # Aplicar padding a la imagen
    if padding > 0:
        padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    else:
        padded_image = image

    # Dimensiones de la imagen después de aplicar padding
    padded_height, padded_width, _ = padded_image.shape

    # Dimensiones de la salida
    output_height = (padded_height - kernel_size) // stride + 1
    output_width = (padded_width - kernel_size) // stride + 1

    # Inicializar imagen de salida
    output_image = np.zeros((output_height, output_width, 3), dtype=np.float32)

    # Invertir el kernel para la convolución
    kernel_flipped = np.flip(kernel)

    # Aplicar la convolución
    for y in range(0, output_height):
        for x in range(0, output_width):
            # Extraer la región de la imagen que corresponde al kernel
            region = padded_image[y*stride:y*stride+kernel_size, x*stride:x*stride+kernel_size]
            
            # Realizar la convolución en los tres canales (BGR)
            for c in range(3):  # Iterar sobre los canales BGR
                output_image[y, x, c] = np.sum(region[:, :, c] * kernel_flipped)

    # Clipping de los valores para mantenerlos en el rango [0, 255]
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    print(f"Kernel aplicado a la imagen {index} correctamente")
    return output_image

def apply_kernel_numpy2(image, kernel, padding=0, stride=1, index=0):
    """
    Applies a kernel over an image using numpy only, with defined padding and stride.
    
    Args:
        image: Input image in BGR format (numpy array).
        kernel: The kernel (matrix) of variable size (3x3, 5x5, etc.).
        padding: Amount of padding to add to the image.
        stride: Number of pixels to move after applying the kernel.
        index: Image index (for tracking in logs).
        
    Returns:
        The convolved image, compatible with Image.fromarray().
    """
    
    print(f"Applying kernel to image {index} using numpy...")
    
    # Ensure kernel is a numpy array
    if isinstance(kernel, list):
        kernel = np.array(kernel, dtype=np.float32)
    
    # Dimensions of the image and kernel
    image_height, image_width, _ = image.shape
    kernel_size = kernel.shape[0]  # Assuming a square kernel

    # Flip kernel for convolution
    kernel_flipped = np.flip(kernel)
    
    # Apply padding to the image if necessary
    if padding > 0:
        padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    else:
        padded_image = image

    # Calculate output dimensions
    output_height = (padded_image.shape[0] - kernel_size) // stride + 1
    output_width = (padded_image.shape[1] - kernel_size) // stride + 1

    # Initialize output image
    output_image = np.zeros((output_height, output_width, 3), dtype=np.float32)

    # Perform convolution manually using numpy for each channel
    for y in range(0, output_height):
        for x in range(0, output_width):
            # Define region of interest
            region = padded_image[y*stride:y*stride+kernel_size, x*stride:x*stride+kernel_size]
            
            # Apply the kernel to each channel (BGR)
            for c in range(3):  # Iterate over channels BGR
                output_image[y, x, c] = np.sum(region[:, :, c] * kernel_flipped)

    # Clip the result to keep it within [0, 255]
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    print(f"Numpy-based kernel applied to image {index} successfully")
    return output_image

def apply_kernel(image, kernel, padding=0, stride=1, index=0):
    """
    Applies a kernel over an image using scipy's correlate, with defined padding and stride.

    Args:
        image: Input image in BGR format (numpy array).
        kernel: The kernel (matrix) of variable size (3x3, 5x5, etc.).
        padding: Amount of padding to add to the image.
        stride: Number of pixels to move after applying the kernel.
        index: Image index (for tracking in logs).

    Returns:
        The convolved image, compatible with Image.fromarray().
    """
    print(f"Applying precise kernel to image {index} using scipy.signal.correlate...")

    # Ensure kernel is a numpy array with float32 type for precision
    kernel = np.array(kernel, dtype=np.float32)

    # Apply padding to the image if necessary
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)

    # Initialize an empty output array
    output_height = (image.shape[0] - kernel.shape[0]) // stride + 1
    output_width = (image.shape[1] - kernel.shape[1]) // stride + 1
    output_image = np.zeros((output_height, output_width, 3), dtype=np.float32)

    # Apply the convolution precisely using correlate on each channel
    for c in range(3):
        output_image[:, :, c] = correlate(image[:, :, c], kernel, mode='valid')[::stride, ::stride]

    # Clip values to ensure they stay within the valid [0, 255] range for images
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    print(f"Precise kernel applied to image {index} successfully")
    return output_image

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
        "rotate_image": rotate_image,
        "apply_colormap": apply_colormap,
        "compress_image": compress_image_dct,
        "resize_image": resize_image,
        "binarize_image": binarize_image
    }
    
    return kernels[name]
# endregion

# region Color transformations
def rgb_to_bgr(image, index=0):
    image_c = image.copy()
    image_c[:, :, ::-1]
    print(f"Imagen {index} convertida a BGR")
    return image_c

def bgr_to_rgb(image, index=0):
    image_c = image.copy()
    image_c[:, :, ::-1]
    print(f"Imagen {index} convertida de BGR a RGB")
    return image_c

def gray_to_rgb(image, index=0):
    result = np.stack((image, image, image), axis=-1)
    print(f"Imagen {index} convertida de Gris a RGB")
    return result

def gray_to_bgr(image, index=0):
    result = np.stack((image, image, image), axis=-1)
    print(f"Imagen {index} convertida de Gris a BGR")
    return result

def rgb_to_gray(image, index=0):
    
    print(f"Convirtiendo imagen {index} a escala de grises...")
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
    
    print(f"Imagen {index} convertida a escala de grises")
    return np.array(gray_image, dtype=np.uint8)

def bgr_to_gray(image, index=0):
    gray_image = (0.1140 * image[:, :, 0]) + (0.5870 * image[:, :, 1]) + (0.2989 * image[:, :, 2])
    print(f"Imagen {index} convertida de BGR a escala de grises")
    return gray_image.astype(np.uint8)

def reduce_red(image, p=0.5, index=0):
    image[:, :, 0] = (image[:, :, 0] * p).astype('uint8')
    print(f"Imagen {index} con el canal rojo reducido")
    return image

def reduce_green(image, p=0.5, index=0):
    image[:, :, 1] = (image[:, :, 1] * p).astype('uint8')
    print(f"Imagen {index} con el canal verde reducido")
    return image

def reduce_blue(image, p=0.5, index=0):
    image[:, :, 2] = (image[:, :, 2] * p).astype('uint8')
    print(f"Imagen {index} con el canal azul reducido")
    return image

def increase_red(image, intensity, index=0):
    output_image = []
    for row in image:
        new_row = []
        for pixel in row:
            r = min(255, int(pixel[0] * (1 + intensity)))
            new_row.append([r, pixel[1], pixel[2]])
        output_image.append(new_row)
    
    print(f"Imagen {index} con el canal rojo aumentado")
    return np.array(output_image, dtype=np.uint8)

def increase_green(image, intensity, index=0):
    output_image = []
    for row in image:
        new_row = []
        for pixel in row:
            g = min(255, int(pixel[1] * (1 + intensity)))
            new_row.append([pixel[0], g, pixel[2]])
        output_image.append(new_row)
        
    print(f"Imagen {index} con el canal verde aumentado")
    return np.array(output_image, dtype=np.uint8)

def increase_blue(image, intensity, index=0):
    output_image = []
    for row in image:
        new_row = []
        for pixel in row:
            b = min(255, int(pixel[2] * (1 + intensity)))
            new_row.append([pixel[0], pixel[1], b])
        output_image.append(new_row)
        
    print(f"Imagen {index} con el canal azul aumentado")
    return np.array(output_image, dtype=np.uint8)
# endregion

# region Image transformations
def plain_image(image, index=0):
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
    
    print(f"Image {index} converted to vector")
    return vector

def normalize_image_vector(image_vector, index=0):
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
    
    print(f"Image {index} vector normalizado")
    return image_vector

def resize_image(image, width, height, interpolation = "Bicubic", index=0):
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
    image_copy = image.copy()
    image_resized = cv2.resize(image_copy, (width, height), interpolation = interpolation)
    
    print(f"Imagen {index} redimencionada a la forma: {image_resized.shape}")
    
    return image_resized

def rotate_image(image, angle, index=0):
    """
    Rotates the input image by the specified angle.
    
    Args:
        image (np.array): Input image as a numpy array.
        angle (float): Angle in degrees to rotate the image.
        
    Returns:
        np.array: Rotated image.
    """
    
    temp_image = image.copy()
     
    # Get image dimensions
    (h, w) = temp_image.shape[:2]
    
    # Find the center of the image
    center = (w // 2, h // 2)
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    rotated_image = cv2.warpAffine(temp_image, M, (w, h))
    
    print(f"Imagen {index} rotada {angle} grados")
    return rotated_image

def apply_colormap(image, colormap="JET", index=0):
    """
    Applies a colormap to a grayscale image.
    
    Args:
        image (np.array): Input grayscale image.
        colormap (int): OpenCV colormap to apply.
        
    Returns:
        np.array: Colormap-applied image.
    """
    
    colormaps = {
        "AUTUMN": cv2.COLORMAP_AUTUMN,
        "BONE": cv2.COLORMAP_BONE,
        "JET": cv2.COLORMAP_JET,
        "WINTER": cv2.COLORMAP_WINTER,
        "RAINBOW": cv2.COLORMAP_RAINBOW,
        "OCEAN": cv2.COLORMAP_OCEAN,
        "SUMMER": cv2.COLORMAP_SUMMER,
        "SPRING": cv2.COLORMAP_SPRING,
        "COOL": cv2.COLORMAP_COOL,
        "HSV": cv2.COLORMAP_HSV,
        "PINK": cv2.COLORMAP_PINK,
        "HOT": cv2.COLORMAP_HOT,
        "PARULA": cv2.COLORMAP_PARULA,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "CIVIDIS": cv2.COLORMAP_CIVIDIS,
        "TWILIGHT": cv2.COLORMAP_TWILIGHT,
        "TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
        "TURBO": cv2.COLORMAP_TURBO,
        "DEEPGREEN": cv2.COLORMAP_DEEPGREEN
    }
    
    if colormap is None or colormap not in colormaps:
        print("Invalid colormap")
        return
    
    colormap_f = colormaps[colormap]
    
    # Check if the image is grayscale; if not, convert it
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply the specified colormap
    colored_image = cv2.applyColorMap(image, colormap_f)
    
    print(f"Mapeo de colores {colormap}  a la imagen {index}")
    return colored_image

def get_histogram_data(image):
    """
    Computes and returns the histogram data of a single image.
    
    Args:
        image (np.array): Input image.
        
    Returns:
        dict: A dictionary containing histograms for each color channel (RGB or grayscale).
    """
    histogram_data = {}

    # Check if the image is colored or grayscale
    if len(image.shape) == 3:  # Color image
        colors = ('r', 'g', 'b')
        for i, col in enumerate(colors):
            # Compute histogram for each channel
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            histogram_data[col] = hist
    else:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram_data['grayscale'] = hist
    
    return histogram_data

def compress_image_dct(image, compression_factor=0.5, index=0):
    """
    Compresses the image using Discrete Cosine Transform (DCT).
    
    Args:
        image (np.array): Input image.
        compression_factor (float): Factor to compress, ranges from 0 (high compression) to 1 (low compression).
        
    Returns:
        np.array: Compressed image.
    """
    # Convert image to grayscale if it's colored
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to float32 for DCT
    image = np.float32(image) / 255.0
    
    # Apply DCT (Discrete Cosine Transform)
    dct = cv2.dct(image)
    
    # Zero out low-value frequencies based on compression factor
    h, w = dct.shape
    compressed_dct = np.zeros((h, w), dtype=np.float32)
    limit_h, limit_w = int(h * compression_factor), int(w * compression_factor)
    
    compressed_dct[:limit_h, :limit_w] = dct[:limit_h, :limit_w]
    
    # Apply inverse DCT to get compressed image
    compressed_image = cv2.idct(compressed_dct)
    
    # Scale back to 8-bit format
    compressed_image = np.uint8(compressed_image * 255)
    
    print(f"Imagen {index} comprimida con factor {compression_factor}")
    return compressed_image

def binarize_image(image, threshold=128, index=0):
    """
    Converts the input image into a binary image (black and white).
    
    Args:
        image (np.array): Input image.
        threshold (int): Threshold value for binarization (0-255).
        
    Returns:
        np.array: Binary image (black and white).
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    print(f"Imagen {index} binarizada con umbral {threshold}")
    return binary_image
# endregion