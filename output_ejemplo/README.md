# Salida de ejemplo

Artefactos de una corrida completa sobre el material de muestra:

```bash
python main.py --input data/raw --output output --method all
```

## Contenido

```
run_metadata.json                  argumentos, dispositivo, versiones, commit, fallos
<video>/method_1/
├── <video>_osd.mp4                video anotado: cajas por nivel de riesgo, pretil, HUD
├── berm_height.{png,svg}          altura en el tiempo con banda de incertidumbre
├── vehicle_spatial.{png,svg}      posiciones en planta y matriz de distancias mínimas
├── berm_profile.csv               serie por frame: altura, cobertura, condición, distancia
├── proximity_events.csv           posición y nivel de riesgo por equipo y frame
├── distance_matrix.csv            distancia mínima por par de equipos
└── metadata.json                  FPS, ms por etapa, alertas, resoluciones, jitter
```

Los cuatro videos están procesados con el **Método 1** (camino óptimo). El **Método 2**
(argmax) está solo en `video_02`, para comparar la estabilidad del perfil entre
`video_02/method_1` y `video_02/method_2`.

## Dos advertencias

**Tres de los cuatro videos son de entrenamiento.** El detector se entrenó con frames de
`video_01`, `video_02` y `video_03`, así que en esos se ve mejor de lo que realmente es.
`video_04` quedó completo para validación y es el único resultado sobre material que el
modelo no vio. Las métricas de detección del reporte se calculan sobre él.

**Los videos se recodificaron.** El pipeline escribe `mp4v`, el único códec que siempre
viene con OpenCV (ver `bermguard/io/video_writer.py`). Acá los pasé a H.264 (CRF 24) para
que quepan en el repositorio: los cinco videos bajan de 56.9 MB a 15.9 MB sin diferencia
visible. El contenido es el que generó el pipeline, y al ejecutar el contenedor se obtienen
los mismos videos en `mp4v`.

## Qué mirar primero

1. **`video_04/method_1/video_04_osd.mp4`**, el único fuera de muestra. En el frame 130, por
   ejemplo, el CAEX y el bulldozer salen como **dos equipos separados** y ambos en
   precaución, algo que el modelo COCO no lograba. Pero la caja del camión es bastante más
   grande que la máquina, y esa caja arrastra la búsqueda del pretil al primer plano: la
   curva queda sobre las huellas de neumático y no sobre el banco detrás del bulldozer. Está
   explicado en el ADR 0007.
2. **`video_01/method_1/berm_height.png`**: la altura con su banda de incertidumbre y los
   huecos en los dos cortes de escena del clip.
3. **`video_02/method_1` contra `video_02/method_2`**: la diferencia de estabilidad, en el
   campo `berm_crest_jitter_px` de cada `metadata.json`.
