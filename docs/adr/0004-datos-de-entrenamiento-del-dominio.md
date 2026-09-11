# ADR 0004: Entrenar con frames del propio material en vez de datasets públicos

- Estado: aceptada
- Fecha: 2026-09-08
- Depende de: ADR 0002

## Contexto

El ADR 0002 decidió entrenar un detector. Faltaba decidir con qué datos.

El enunciado sugiere datasets públicos y nombra Roboflow Universe. Evalué tres, fijándome
sobre todo en qué tan parecidos son a este material, porque lo que hizo fallar a YOLO-World
fue justamente la distancia entre sus datos de entrenamiento y estos videos.

| Dataset | Clases | Imágenes | Licencia | Problema |
|---|---|---|---|---|
| `brancos-workspace/mining-truck-detection-w53tl` | 1 (`haul_truck`) | 264 | CC BY 4.0 | No tiene bulldozer. Solo aporta la clase que el modelo base ya detecta, y parte de las imágenes son aumentos de un mismo frame |
| `mohamads-workspace-rl4ui/bulldozer-lpoe2` | 7, incluye `Bull_dozer` | 1.324 | CC BY 4.0 | Tiene las clases, pero son fotos de catálogo: la máquina llena el encuadre, bien iluminada y sin oclusión |
| `rf100-vl/apoce-...-construction-equipment` | 7, incluye `bulldozer` | 928 | MIT | Fotos cenitales de dron sobre obras urbanas, con objetos muy pequeños |

Ninguno se parece a estos videos: vista oblicua desde altura, distancia media, desierto con
polvo y más de la mitad de los frames de noche.

## Decisión

**Entrenar con frames anotados del propio material.** Los extraje con
`scripts/prepare_labeling_set.py`, estratificados por condición de luz, y anoté 171.

Creo que ese volumen alcanza por dos razones:

- El entrenamiento parte de pesos preentrenados, que ya saben de bordes, texturas y
  vehículos. Lo que falta enseñar es cómo se ven estas dos máquinas desde esta cámara.
- El material es homogéneo: misma faena, mismas máquinas y encuadres parecidos. En otro
  contexto sería un riesgo de sobreajuste, pero lo más probable es que el conjunto de
  evaluación venga del mismo lugar.

**Las clases son `caex` y `bulldozer`**, que son roles en el botadero. Los datasets
públicos etiquetan `truck` o `vehicle`, que no es la distinción que necesito.

## Alternativas descartadas

- **`bulldozer-lpoe2` como fuente principal.** Enseñaría a esperar máquinas grandes,
  centradas y bien iluminadas, lo contrario de este material. Queda como experimento:
  preentrenar con él y afinar con los frames propios.
- **APOCE.** Una vista cenital y una oblicua dan siluetas distintas de la misma máquina.
- **Juntar los tres datasets sin datos propios.** Más volumen no corrige el sesgo: 2.500
  imágenes fuera de dominio siguen fuera de dominio.

## Consecuencias

- Anotar me tomó cerca de dos horas.
- El modelo queda muy especializado y no va a generalizar a otras faenas ni a otros
  ángulos de cámara.
- La validación **no puede ser aleatoria**. Frames del mismo video son casi iguales, y una
  partición aleatoria mediría memoria y no generalización. Separé por video y dejé
  `video_04` completo para validación.
