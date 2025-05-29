import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from shapely.geometry import Point, box
from shapely.affinity import scale, rotate, translate
from shapely.strtree import STRtree
import random
import shapely
import os
import time
from tqdm import tqdm

inicio=time.time()

numero_imagenes=1


def generate_fiber_image(
    b_ini=20.5,
    b_end=200,
    width=510,  # Ancho de la imagen en unidades arbitrarias
    height=310,  # Alto de la imagen en unidades arbitrarias
    a=20,
    phi_range=(0, 90),
    p=0.30,  # Porcentaje de área objetivo
    max_attempts=20000,
):
    fig, ax = plt.subplots()

    # Establecer el color de fondo de la figura y los ejes a negro
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Configurar los límites de los ejes para que coincidan con las dimensiones de la imagen
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")  # Ocultar los ejes

    # Asegurar que la escala sea igual en ambos ejes
    ax.set_aspect("equal")

    # Eliminar márgenes adicionales
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.margins(0)

    # Crear un cuadro que representa los límites de la imagen
    image_bounds = box(0, 0, width, height)

    target_area = width * height * p
    current_area = 0
    ellipses = []
    ellipse_shapes = []

    attempts = 0
    N = 0  # Contador de elipses generadas
    sum_sin2_beta = 0
    sum_cos2_beta_sin2_phi = 0
    sum_cos2_beta_cos2_phi = 0

    # Calcular el máximo tamaño de 'a' y 'b' para generar posiciones adecuadas
    max_a = a + 0.5
    max_b = b_end

    while current_area < target_area and attempts < max_attempts:
        # Generar un 'a' aleatorio dentro de la tolerancia de ±0.5
        a_random = random.uniform(a - 0.5, a + 0.5)
        # Generar un 'b' aleatorio entre 'b_ini' y 'b_end'
        b_random = random.uniform(b_ini, b_end)

        # Asegurarnos de que 'a_random' y 'b_random' sean positivos y 'a_random' <= 'b_random'
        a_random = max(
            0.1, min(a_random, width * 2)
        )  # Ajuste para evitar restricciones
        b_random = max(
            a_random, min(b_random, height * 2)
        )  # Asegurar que b_random >= a_random

        # Generar posición aleatoria permitiendo que las elipses se extiendan más allá de los bordes
        x = random.uniform(-max_b, width + max_b)
        y = random.uniform(-max_b, height + max_b)
        phi = random.uniform(*phi_range)

        # Crear una elipse como objeto shapely con mayor resolución
        ellipse_shape = Point(0, 0).buffer(1, resolution=128)
        ellipse_shape = scale(ellipse_shape, a_random / 2, b_random / 2)
        ellipse_shape = rotate(ellipse_shape, phi, use_radians=False)
        ellipse_shape = translate(ellipse_shape, x, y)

        # Verificar solapamiento utilizando STRtree
        overlap = False
        if ellipse_shapes:
            # Construir el STRtree con las formas existentes
            tree = STRtree(ellipse_shapes)

            # Dependiendo de la versión de Shapely, 'query' devuelve índices o geometrías
            if shapely.__version__ >= "2.0.0":
                # En Shapely 2.0.0+, 'query' devuelve índices
                possible_overlap_indices = tree.query(ellipse_shape)
                # Obtener las geometrías correspondientes
                possible_overlaps = [
                    ellipse_shapes[i] for i in possible_overlap_indices
                ]
            else:
                # En versiones anteriores, 'query' devuelve geometrías directamente
                possible_overlaps = tree.query(ellipse_shape)

            for other in possible_overlaps:
                if ellipse_shape.intersects(other):
                    overlap = True
                    break

        if not overlap:
            # Calcular el área de la elipse que está dentro de la imagen
            intersection_area = ellipse_shape.intersection(image_bounds).area
            if intersection_area > 0:
                current_area += intersection_area
                N += 1  # Incrementar contador de elipses

                # Calcular beta_i
                sin_beta_i = a_random / b_random
                # Asegurar que sin_beta_i no exceda 1 debido a errores numéricos
                sin_beta_i = min(1.0, sin_beta_i)
                beta_i = np.arcsin(sin_beta_i)

                # Calcular sin^2 beta_i y cos^2 beta_i
                sin2_beta_i = np.sin(beta_i) ** 2
                cos2_beta_i = np.cos(beta_i) ** 2

                # Convertir phi a radianes para funciones trigonométricas
                phi_i_rad = np.deg2rad(phi)

                # Calcular sin^2 phi_i y cos^2 phi_i
                sin2_phi_i = np.sin(phi_i_rad) ** 2
                cos2_phi_i = np.cos(phi_i_rad) ** 2

                # Acumular las sumas para el tensor de orientación
                sum_sin2_beta += sin2_beta_i
                sum_cos2_beta_sin2_phi += cos2_beta_i * sin2_phi_i
                sum_cos2_beta_cos2_phi += cos2_beta_i * cos2_phi_i

            ellipses.append({"x": x, "y": y, "a": a_random, "b": b_random, "phi": phi})
            ellipse_shapes.append(ellipse_shape)

        attempts += 1

    # Calcular los componentes del tensor de orientación
    if N > 0:
        a_xx = sum_sin2_beta / N
        a_yy = sum_cos2_beta_sin2_phi / N
        a_zz = sum_cos2_beta_cos2_phi / N
    else:
        a_xx = a_yy = a_zz = 0

    # Mostrar los resultados
    # print(f"Tensor de orientación:")
    # print(f"a_xx = {a_xx:.4f}")
    # print(f"a_yy = {a_yy:.4f}")
    # print(f"a_zz = {a_zz:.4f}")

    # Dibujar las elipses con colores que contrasten con el fondo negro
    for e in ellipses:
        x, y, a_random, b_random, phi = e["x"], e["y"], e["a"], e["b"], e["phi"]
        ellipse = Ellipse(
            (x, y),
            width=a_random,
            height=b_random,
            angle=phi,
            edgecolor="white",  # Blanco para que sea visible sobre negro
            facecolor="white",  # Blanco para que sea visible sobre negro
            linewidth=0,  # Eliminar bordes para evitar que se vean más gruesos
        )
        ax.add_patch(ellipse)

    plt.gca().invert_yaxis()

    # Guardar la imagen sin márgenes adicionales en la carpeta images:

    # Crear la carpeta 'images' si no existe
    if not os.path.exists('images'):
        os.makedirs('images')

    # Guardar la imagen en la carpeta 'images'
    plt.savefig(
        f"images_try/{a_xx:.4f},{a_yy:.4f},{a_zz:.4f}.jpg",
        bbox_inches="tight",
        pad_inches=0,
        facecolor="black",
    )
    plt.show()
    # print(a_xx,a_yy,a_zz)
    plt.close(fig)

    # Si prefieres mostrar la imagen directamente
    # plt.show()


# Ejecutar la función
for i in tqdm(range(numero_imagenes)):
    generate_fiber_image()
final=time.time()
print(f'{numero_imagenes} imágenes generadas en {(final-inicio)/60:.2f} minutos')
