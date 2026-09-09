# BermGuard AI — imagen de ejecucion
#
# Base CUDA segun ADR 0001. La imagen se construye sobre el runtime de NVIDIA y
# funciona igualmente en una maquina sin GPU: en ese caso torch.cuda.is_available()
# devuelve False y el pipeline degrada a CPU registrando una advertencia. Eso es
# deliberado, porque el comando de referencia del enunciado no pasa --gpus.
#
# Ubuntu 22.04 trae Python 3.10 de sistema, que es la misma version del entorno de
# desarrollo. Esa paridad evita la clase de fallo mas molesta de reproducir: la que
# solo aparece dentro del contenedor.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Sin buffer en stdout: los logs de un contenedor deben aparecer cuando ocurren, no
# cuando el buffer se llena. Sin .pyc: no aportan nada en una imagen inmutable.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Ultralytics escribe configuracion y telemetria en el home del usuario. Se le
    # fija un directorio propio y escribible, y se desactiva el envio de analitica.
    YOLO_CONFIG_DIR=/app/.ultralytics \
    MPLCONFIGDIR=/app/.matplotlib

# libgl1 y libglib2.0-0 son dependencias de sistema de OpenCV. Sin ellas
# `import cv2` falla en tiempo de ejecucion aunque pip haya instalado bien: es
# exactamente el tipo de problema que un entorno virtual no puede resolver.
# ffmpeg aporta los codecs para decodificar los videos de entrada.
#
# python-is-python3 no es opcional: Ubuntu 22.04 provee `python3` pero no `python`,
# y el comando de referencia del enunciado invoca literalmente `python main.py`.
# Sin este paquete la imagen falla con "python: command not found" en la primera
# ejecucion del evaluador.
RUN apt-get update && apt-get install --no-install-recommends -y \
        python3.10 \
        python3-pip \
        python-is-python3 \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Las dependencias se copian e instalan antes que el codigo. Es la capa mas pesada
# y la que menos cambia, de modo que este orden permite reconstruir tras editar el
# codigo sin reinstalar torch.
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0 \
    && python3 -m pip install -r requirements.txt

# Los pesos van horneados en la imagen, no se descargan al arrancar. El enunciado
# exige ejecucion "out-of-the-box": el entorno de evaluacion puede no tener red, y
# una descarga en el primer arranque haria la imagen lenta, no determinista y
# dependiente de que el servidor de origen siga en pie.
COPY weights/ ./weights/

COPY configs/ ./configs/
COPY bermguard/ ./bermguard/
COPY main.py pyproject.toml ./

RUN python3 -m pip install --no-deps -e . \
    && mkdir -p /app/test /app/output /app/.ultralytics /app/.matplotlib

# Usuario sin privilegios. Un contenedor que escribe en volumenes montados no
# necesita root, y correr como root deja los archivos de salida pertenecientes a
# root en la maquina del evaluador.
RUN useradd --create-home --uid 1000 bermguard \
    && chown -R bermguard:bermguard /app
USER bermguard

# ENTRYPOINT vacio a proposito. El comando del enunciado pasa `python main.py ...`
# como argumentos, y un ENTRYPOINT ["python","main.py"] los duplicaria y romperia
# la invocacion literal que se va a evaluar.
ENTRYPOINT []
CMD ["python", "main.py", "--input", "/app/test", "--output", "/app/output", "--method", "all"]
