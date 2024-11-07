import serial
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

# Configura a porta serial (ajuste conforme necessário)
ser = serial.Serial('COM5', 115200, timeout=1)

# Defina os tamanhos conforme o seu código C++
EI_CLASSIFIER_INPUT_WIDTH = 64  # ajuste conforme necessário
EI_CLASSIFIER_INPUT_HEIGHT = 64  # ajuste conforme necessário
EI_CLASSIFIER_LABEL_COUNT = 2  # ajuste conforme necessário

# Tamanho da imagem em RGB888
IMAGE_SIZE = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3

# Função para ler os dados da serial
def read_serial_data():

    while ser.in_waiting < IMAGE_SIZE + EI_CLASSIFIER_LABEL_COUNT * 4:
        print("Waiting for data..." + str(ser.in_waiting))
        time.sleep(0.1)  # Aguarde um pouco antes de verificar novamente

    # Leia a imagem
    image_data = ser.read(IMAGE_SIZE)  # Leia a imagem em big-endian
    #print("Image data: ", image_data)
    
    # Leia os resultados da classificação
    classification_data = ser.read(EI_CLASSIFIER_LABEL_COUNT * 4)  # 4 bytes por float
    
    # Converta os dados da imagem para um array numpy
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image_array = image_array.reshape((EI_CLASSIFIER_INPUT_HEIGHT, EI_CLASSIFIER_INPUT_WIDTH, 3))
    
    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    hist_image = bgr_image.copy()
     # Equalize o histograma de cada canal de cor
    hist_image[:, :, 0] = cv2.equalizeHist(image_array[:, :, 0])
    hist_image[:, :, 1] = cv2.equalizeHist(image_array[:, :, 1])
    hist_image[:, :, 2] = cv2.equalizeHist(image_array[:, :, 2])

    # Converta os dados de classificação para floats
    classification_results = struct.unpack('f' * EI_CLASSIFIER_LABEL_COUNT, classification_data)
    
    return hist_image, classification_results

# Função para exibir a imagem e os resultados da classificação
def display_data(image_array, classification_results):
    plt.imshow(image_array)
    plt.title(f'Classification Results: {classification_results}')

# Loop principal
while True:
    image_array, classification_results = read_serial_data()
    display_data(image_array, classification_results)
    plt.pause(0.1)  # Aguarde um pouco antes de exibir a próxima imagem