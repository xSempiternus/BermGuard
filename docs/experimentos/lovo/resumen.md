### Validación cruzada dejando un video fuera (LOVO)

**Pregunta.** ¿Alguna variante (copy-paste de bulldozers, preentrenamiento con datos públicos) detecta mejor el bulldozer que `base` en un video que no vio, sin empeorar el CAEX? Y, para la ganadora, ¿qué umbral de bulldozer usar?

**Datos.** Un entrenamiento por variante y video excluido; cada fold se evalúa en el video que no vio (`docs/experimentos/lovo/{variante}_{video}.json`). Media ± desviación estándar muestral entre folds; precisión, recall y F1 en el umbral de operación de cada JSON.

Reproducir: `python scripts/summarize_lovo.py`

#### `bulldozer`: media ± desviación entre folds

| Variante | Folds | AP@0.5 | AP@0.5:0.95 | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|
| `base` | 4 | 0.751 ± 0.240 | 0.557 ± 0.266 | 0.811 ± 0.244 | 0.750 ± 0.260 | 0.756 ± 0.228 |
| `copypaste` | 4 | 0.767 ± 0.222 | 0.582 ± 0.211 | 0.756 ± 0.122 | 0.766 ± 0.209 | 0.751 ± 0.145 |
| `pretrain` | 4 | 0.825 ± 0.168 | 0.586 ± 0.156 | 0.797 ± 0.182 | 0.796 ± 0.247 | 0.773 ± 0.185 |

Detalle por fold de `bulldozer` (objetos entre paréntesis):

| Variante | Métrica | video_01 | video_02 | video_03 | video_04 |
|---|---|---|---|---|---|
| `base` | AP@0.5 | 0.559 (27) | 0.912 (14) | 1.000 (17) | 0.532 (16) |
| `base` | AP@0.5:0.95 | 0.375 (27) | 0.737 (14) | 0.829 (17) | 0.286 (16) |
| `base` | Recall | 0.444 (27) | 0.929 (14) | 1.000 (17) | 0.625 (16) |
| `copypaste` | AP@0.5 | 0.596 (27) | 0.912 (14) | 1.000 (17) | 0.559 (16) |
| `copypaste` | AP@0.5:0.95 | 0.397 (27) | 0.722 (14) | 0.804 (17) | 0.407 (16) |
| `copypaste` | Recall | 0.518 (27) | 0.857 (14) | 1.000 (17) | 0.688 (16) |
| `pretrain` | AP@0.5 | 0.624 (27) | 0.921 (14) | 1.000 (17) | 0.756 (16) |
| `pretrain` | AP@0.5:0.95 | 0.432 (27) | 0.702 (14) | 0.738 (17) | 0.472 (16) |
| `pretrain` | Recall | 0.444 (27) | 0.929 (14) | 1.000 (17) | 0.812 (16) |

#### `caex`: media ± desviación entre folds

| Variante | Folds | AP@0.5 | AP@0.5:0.95 | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|
| `base` | 4 | 0.738 ± 0.232 | 0.522 ± 0.213 | 0.782 ± 0.080 | 0.688 ± 0.304 | 0.695 ± 0.207 |
| `copypaste` | 4 | 0.770 ± 0.154 | 0.561 ± 0.149 | 0.801 ± 0.070 | 0.703 ± 0.240 | 0.724 ± 0.154 |
| `pretrain` | 4 | 0.771 ± 0.196 | 0.564 ± 0.188 | 0.806 ± 0.094 | 0.744 ± 0.268 | 0.742 ± 0.164 |

Detalle por fold de `caex` (objetos entre paréntesis):

| Variante | Métrica | video_01 | video_02 | video_03 | video_04 |
|---|---|---|---|---|---|
| `base` | AP@0.5 | 0.437 (170) | 0.957 (33) | 0.678 (43) | 0.880 (34) |
| `base` | AP@0.5:0.95 | 0.227 (170) | 0.683 (33) | 0.504 (43) | 0.674 (34) |
| `base` | Recall | 0.277 (170) | 0.970 (33) | 0.651 (43) | 0.853 (34) |
| `copypaste` | AP@0.5 | 0.555 (170) | 0.882 (33) | 0.761 (43) | 0.881 (34) |
| `copypaste` | AP@0.5:0.95 | 0.355 (170) | 0.665 (33) | 0.547 (43) | 0.676 (34) |
| `copypaste` | Recall | 0.359 (170) | 0.879 (33) | 0.721 (43) | 0.853 (34) |
| `pretrain` | AP@0.5 | 0.503 (170) | 0.955 (33) | 0.763 (43) | 0.864 (34) |
| `pretrain` | AP@0.5:0.95 | 0.295 (170) | 0.714 (33) | 0.578 (43) | 0.669 (34) |
| `pretrain` | Recall | 0.359 (170) | 0.939 (33) | 0.767 (43) | 0.912 (34) |

#### Diferencia pareada contra `base` (variante − base, por fold)

| Variante | Clase | Métrica | Media | Mín | Máx | Folds mejor/peor/empate | p signos | Veredicto |
|---|---|---|---|---|---|---|---|---|
| `copypaste` | `bulldozer` | AP@0.5 | +0.016 | +0.000 | +0.036 | 2/0/2 de 4 | 0.500 | no distinguible del ruido |
| `copypaste` | `bulldozer` | AP@0.5:0.95 | +0.025 | -0.026 | +0.121 | 2/2/0 de 4 | 1.000 | no distinguible del ruido |
| `copypaste` | `bulldozer` | Recall | +0.016 | -0.072 | +0.074 | 2/1/1 de 4 | 1.000 | no distinguible del ruido |
| `copypaste` | `caex` | AP@0.5 | +0.032 | -0.074 | +0.118 | 2/1/1 de 4 | 1.000 | no distinguible del ruido |
| `copypaste` | `caex` | AP@0.5:0.95 | +0.039 | -0.017 | +0.128 | 2/1/1 de 4 | 1.000 | no distinguible del ruido |
| `copypaste` | `caex` | Recall | +0.015 | -0.091 | +0.082 | 2/1/1 de 4 | 1.000 | no distinguible del ruido |
| `pretrain` | `bulldozer` | AP@0.5 | +0.075 | +0.000 | +0.224 | 2/0/2 de 4 | 0.500 | no distinguible del ruido |
| `pretrain` | `bulldozer` | AP@0.5:0.95 | +0.029 | -0.092 | +0.186 | 2/2/0 de 4 | 1.000 | no distinguible del ruido |
| `pretrain` | `bulldozer` | Recall | +0.047 | +0.000 | +0.188 | 1/0/3 de 4 | 1.000 | no distinguible del ruido |
| `pretrain` | `caex` | AP@0.5 | +0.033 | -0.016 | +0.085 | 2/1/1 de 4 | 1.000 | no distinguible del ruido |
| `pretrain` | `caex` | AP@0.5:0.95 | +0.042 | -0.006 | +0.074 | 3/0/1 de 4 | 0.250 | mejora |
| `pretrain` | `caex` | Recall | +0.057 | -0.030 | +0.116 | 3/1/0 de 4 | 0.625 | mejora |

Criterio: un fold mejora si Δ > 0.01; la métrica mejora si lo hacen al menos 3 folds y la media es ≥ 0.02. Con 4 folds ni 4 de 4 es significativo al 5 % (p de signos bilateral = 0.125): es un criterio de consistencia, no una prueba.

#### Decisión

**Variante elegida: `base`.** Ninguna variante cumple la regla: base por parsimonia.
- `copypaste`: descartada: AP@0.5 bulldozer: no distinguible del ruido; AP@0.5 caex cae más de 0.05 en algún fold.
- `pretrain`: descartada: AP@0.5 bulldozer: no distinguible del ruido.

#### Umbral de `bulldozer` fuera de fold (`base`)

Criterio: `precision_min` con mínimo 0.5. Para cada fold el umbral se elige con los barridos de los otros tres y se mide en el que queda fuera.

| Fold | Umbral | Elegido con | Precisión | Recall | F1 | Objetos |
|---|---|---|---|---|---|---|
| video_01 | 0.01 (borde de la grilla) | video_02, video_03, video_04 | 0.395 | 0.556 | 0.462 | 27 |
| video_02 | 0.03 | video_01, video_03, video_04 | 0.722 | 0.929 | 0.813 | 14 |
| video_03 | 0.03 | video_01, video_02, video_04 | 0.810 | 1.000 | 0.895 | 17 |
| video_04 | 0.03 | video_01, video_02, video_03 | 0.387 | 0.750 | 0.511 | 16 |
| media ± desv. | | | 0.578 ± 0.219 | 0.809 ± 0.199 | 0.670 ± 0.216 | |

Sumando los cuatro folds, cada uno con su umbral fuera de fold: precisión 0.528, recall 0.770 (57 de 74 objetos).

**Umbral para producción**, elegido con los cuatro folds: **0.03**. Su rendimiento esperado es el fuera de fold de arriba. En `configs/`: `detector.class_confidence: {bulldozer: 0.03}`.

**Qué no se concluye.** Cuatro videos generados no son una muestra de minas; una diferencia marcada como ruido no es evidencia de que las variantes sean iguales, sólo de que estos datos no las distinguen.
