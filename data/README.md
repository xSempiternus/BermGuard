# Datos

Los videos e imágenes de origen **no se versionan** en este repositorio: no son material
propio y pesan más de lo razonable para git.

## Estructura

```
data/
├── raw/     # videos e imágenes de origen (ignorado por git)
└── gt/      # ground truth anotado a mano para el benchmark (sí se versiona)
```

## Cómo reproducir el entorno de datos

1. Copiar los videos de origen en `data/raw/`.
2. Verificar que se leen correctamente:

   ```bash
   python -m bermguard.io.video_reader --probe data/raw
   ```

El pipeline nunca lee de `data/` en producción: recibe la carpeta de entrada por
`--input`, tal como se documenta en el `README.md` raíz.

## Ground truth (`data/gt/`)

Frames anotados manualmente para calcular las métricas del benchmark. Se versionan para
que la evaluación sea reproducible por terceros sin repetir el proceso de anotación.
El detalle del muestreo, el procedimiento y sus limitaciones están en
`reporte_benchmark.md`.
