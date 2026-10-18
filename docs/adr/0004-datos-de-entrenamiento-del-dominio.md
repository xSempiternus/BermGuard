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

## Revisión (2026-10-18): preentrenar con datos públicos, como variante medida

Volví a buscar datasets públicos, ahora para el experimento que quedó abierto: preentrenar
con datos públicos y afinar con los frames propios. Revisé la vista previa de cada uno:

| Dataset | Clases útiles | Imágenes | Vista | Parecido con este material |
|---|---|---|---|---|
| [`ccl-kl4nk/dji_65`](https://universe.roboflow.com/ccl-kl4nk/dji_65) (CC0) | `dozer`, `tippertruck` | 136 | Dron **oblicuo** sobre una mina a cielo abierto, con polvo | El más cercano en geometría y escena. Pocas imágenes y objetos chicos; tolvas de carretera, no CAEX |
| [`bulldozer-lpoe2`](https://universe.roboflow.com/mohamads-workspace-rl4ui/bulldozer-lpoe2) (CC BY 4.0) | `Bull_dozer` | 1.324 | Fotos de catálogo | Aporta la forma del bulldozer, no la escena |
| [`minitruck/mining-truck-v1.4`](https://universe.roboflow.com/minitruck/mining-truck-v1.4) | `mining-truck` | 430 | Cámara fija a nivel del suelo, CAEX en primer plano | Sólo CAEX y a otra escala |
| [`igensol/mines-vehicles`](https://universe.roboflow.com/igensol/mines-vehicles) | `truck` | 422 | CCTV nocturno con polvo en una mina | Las condiciones nocturnas sí; las máquinas no |
| [mineCareerDB](https://data.mendeley.com/datasets/c5s76mj4bm/5) (Mendeley, 2024) | `trucks`, equipos | 203 | Dron a ~400 m | Cenital y sin clase bulldozer |

Sigue sin existir un dataset con este encuadre, y la conclusión sobre la fuente principal
no cambia. Pero el preentrenamiento no necesita el mismo dominio: sirve si le enseña al
modelo la forma de un bulldozer, que es la clase con 33 instancias propias. La pregunta es
empírica, así que queda como una tercera variante de la validación cruzada del ADR 0009:

- `scripts/prepare_pretrain.py` junta los exports YOLO de Roboflow y traduce las clases a
  `caex` y `bulldozer` (`bermguard/training/pretrain.py`). Las clases sin traducción, como
  `excavator`, se descartan: forzarlas a una de las dos enseñaría una confusión.
- `scripts/cross_validate.py --variants pretrain` preentrena una sola vez y afina cada fold
  con la misma partición e hiperparámetros que `base`. Lo único que cambia es el punto de
  partida, así que la diferencia con `base` mide el preentrenamiento. Como ninguna imagen
  pública viene de estos videos, no hay fuga entre folds.

Pasos para correrla, con la GPU libre:

1. En cada página de Roboflow: *Download Dataset* → formato **YOLOv8** → *download zip*
   (pide una cuenta gratuita). Descomprimir en `data/externos/<nombre>/`.
2. Revisar `data.yaml` de cada uno y ajustar el mapa de clases si los nombres difieren:

   ```bash
   python scripts/prepare_pretrain.py        --source data/externos/dji_65 --source data/externos/bulldozer-lpoe2        --map dozer=bulldozer bull_dozer=bulldozer tippertruck=caex
   ```

   El script imprime, por fuente, cuántas instancias quedaron de cada clase y qué clases
   descartó; el resumen queda en `data/pretrain/resumen.json`.
3. `python scripts/cross_validate.py --variants base copypaste pretrain --batch 4`.

Recomiendo empezar sólo con `dji_65` y `bulldozer-lpoe2`: son los que tienen bulldozer, la
clase que falta. Los de CAEX aportan poco, porque el CAEX ya se detecta bien.

Los datasets descargados quedan fuera del repo (`data/externos/`, `data/pretrain/`).
