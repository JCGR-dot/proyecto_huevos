"""
API de inferencia para el proyecto de detección de huevos dañados (CNN + política de rutas).

Sirve el modelo exportado por el notebook (modelo_huevos_cnn.keras / mejor_modelo_huevos.keras)
y config_banda.json (umbrales T_CRACK / CONF_MIN, nombres de clase y ruta por clase).

Ejecutar:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import base64
import io
import json
import time
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pydantic import BaseModel

import tensorflow as tf
from tensorflow import keras

BASE_DIR = Path(__file__).parent
MODELO_DIR = BASE_DIR / "modelo"

# ---------------------------------------------------------------------------
# Carga de modelo y configuración (una sola vez, al arrancar el servidor)
# ---------------------------------------------------------------------------

def _buscar_modelo() -> Path:
    candidatos = ["modelo_huevos_cnn.keras", "mejor_modelo_huevos.keras"]
    for nombre in candidatos:
        ruta = MODELO_DIR / nombre
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        f"No se encontró el modelo en {MODELO_DIR} (se esperaba uno de {candidatos}). "
        "Copia ahí el .keras que exporta la celda 'Exportación del modelo' del notebook."
    )


def _cargar_config() -> dict:
    ruta = MODELO_DIR / "config_banda.json"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró config_banda.json en {MODELO_DIR}. "
            "Cópialo desde la salida del notebook."
        )
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = _cargar_config()
CLASS_NAMES = CONFIG["class_names"]
IMG_SIZE = int(CONFIG["img_size"])
T_CRACK = float(CONFIG["t_crack"])
CONF_MIN = float(CONFIG["conf_min"])
RUTA_POR_CLASE = CONFIG["ruta_por_clase"]  # {"Crack": "...", "Fertile": "...", "Infertile": "..."}

IDX_CRACK = next(i for i, c in enumerate(CLASS_NAMES) if c.lower() == "crack")

MODELO_PATH = _buscar_modelo()
modelo_cnn = keras.models.load_model(MODELO_PATH)
modelo_cnn(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"), training=False)  # warm-up

# Capa convolucional para Grad-CAM: se intenta "relu_4b" (nombre usado en el notebook);
# si no existe (arquitectura distinta), se usa la última capa con salida 4D.
def _encontrar_capa_gradcam():
    if "relu_4b" in [l.name for l in modelo_cnn.layers]:
        return modelo_cnn.get_layer("relu_4b")
    for capa in reversed(modelo_cnn.layers):
        if len(capa.output.shape) == 4:
            return capa
    return None


_CAPA_GRADCAM = _encontrar_capa_gradcam()
_modelo_gradcam = None
if _CAPA_GRADCAM is not None:
    _modelo_gradcam = keras.Model(modelo_cnn.inputs, [_CAPA_GRADCAM.output, modelo_cnn.output])

_LOCK = Lock()  # TensorFlow/Keras no es thread-safe entre requests concurrentes

# ---------------------------------------------------------------------------
# Lógica de negocio (idéntica a la del notebook)
# ---------------------------------------------------------------------------

def decidir_ruta(p: np.ndarray) -> str:
    if p[IDX_CRACK] >= T_CRACK:
        return "Descarte"
    resto = p.copy()
    resto[IDX_CRACK] = 0
    clase = int(resto.argmax())
    if p[clase] < CONF_MIN:
        return "Revisión manual"
    return RUTA_POR_CLASE[CLASS_NAMES[clase]]


def preprocesar(imagen_pil: Image.Image) -> np.ndarray:
    """Igual que la demo interactiva (Gradio) del notebook: sin recorte por caja
    (no hay detector de objetos en producción), solo relleno a cuadrado + resize."""
    img = ImageOps.pad(imagen_pil.convert("RGB"), (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
    return np.asarray(img, dtype="float32")


def gradcam(img_float: np.ndarray, clase: int) -> np.ndarray:
    x = tf.convert_to_tensor(img_float[None])
    with tf.GradientTape() as tape:
        mapas, pred = _modelo_gradcam(x, training=False)
        puntaje = pred[:, clase]
    grads = tape.gradient(puntaje, mapas)
    pesos_canal = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(mapas[0] * pesos_canal, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]


def _jet(valor: np.ndarray) -> np.ndarray:
    """Colormap tipo 'jet' sin depender de matplotlib, para no cargarlo en el servidor."""
    v = np.clip(valor, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def imagen_a_base64(arr_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(title="API clasificador de huevos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PeticionPrediccion(BaseModel):
    imagen: str  # base64, con o sin encabezado "data:image/jpeg;base64,"
    incluir_gradcam: Optional[bool] = False


class RespuestaPrediccion(BaseModel):
    clase: str
    probabilidades: dict
    ruta: str
    latencia_ms: float
    gradcam: Optional[str] = None


def _decodificar_base64(imagen_b64: str) -> Image.Image:
    if "," in imagen_b64 and imagen_b64.strip().startswith("data:"):
        imagen_b64 = imagen_b64.split(",", 1)[1]
    try:
        datos = base64.b64decode(imagen_b64)
        return Image.open(io.BytesIO(datos))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo decodificar la imagen: {exc}")


@app.get("/salud")
def salud():
    return {
        "status": "ok",
        "clases": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "t_crack": T_CRACK,
        "conf_min": CONF_MIN,
        "gradcam_disponible": _modelo_gradcam is not None,
    }


@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(peticion: PeticionPrediccion):
    t0 = time.time()
    imagen_pil = _decodificar_base64(peticion.imagen)
    arr = preprocesar(imagen_pil)

    with _LOCK:
        probs = modelo_cnn(arr[None], training=False).numpy()[0]
        clase_idx = int(probs.argmax())
        ruta = decidir_ruta(probs)

        gradcam_b64 = None
        if peticion.incluir_gradcam and _modelo_gradcam is not None:
            cam = gradcam(arr, clase_idx)
            overlay = (0.55 * arr + 0.45 * _jet(cam) * 255).astype("uint8")
            gradcam_b64 = imagen_a_base64(overlay)

    return RespuestaPrediccion(
        clase=CLASS_NAMES[clase_idx],
        probabilidades={c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        ruta=ruta,
        latencia_ms=round((time.time() - t0) * 1000, 1),
        gradcam=gradcam_b64,
    )
