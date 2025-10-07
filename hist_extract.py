# hist_image.py

import cv2
import matplotlib.pyplot as plt
import sys

def plot_histogram(image_path):
    """
    Carrega uma imagem, calcula o histograma de cores para cada canal (Azul, Verde, Vermelho)
    e exibe o resultado em um gráfico.
    """
    # Tenta carregar a imagem a partir do caminho fornecido
    image = cv2.imread(image_path)
    
    # Verifica se a imagem foi carregada com sucesso. Se não, exibe um erro.
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem do caminho: {image_path}")
        return

    # Separa os canais de cores (OpenCV carrega imagens em formato BGR por padrão)
    colors = ('b', 'g', 'r')
    
    # Cria uma figura para o plot
    plt.figure()
    plt.title('Histograma de Cores')
    plt.xlabel('Intensidade do Pixel (Bins)')
    plt.ylabel('Número de Pixels')

    # Itera sobre cada canal de cor (b, g, r) para calcular e plotar o histograma
    for i, color in enumerate(colors):
        # cv2.calcHist(images, channels, mask, histSize, ranges)
        # images: a imagem de entrada como uma lista [image]
        # channels: o índice do canal para o qual calcular o histograma [i]
        # mask: máscara para calcular o histograma em uma região específica (None para a imagem inteira)
        # histSize: número de bins (níveis de intensidade), 256 para o espectro completo de 0 a 255
        # ranges: o intervalo de valores dos pixels [0, 256]
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        
        # Plota o histograma do canal atual
        plt.plot(hist, color=color)
        plt.xlim([0, 256]) # Define o limite do eixo X

    # Exibe o gráfico do histograma
    plt.show()

if __name__ == '__main__':
    # Verifica se o caminho da imagem foi fornecido como argumento na linha de comando
    if len(sys.argv) != 2:
        print("Uso correto: python3 hist_image.py <caminho_para_sua_imagem>")
    else:
        # Pega o caminho da imagem do argumento da linha de comando
        image_path = sys.argv[1]
        # Chama a função para gerar e mostrar o histograma
        plot_histogram(image_path)