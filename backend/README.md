# Backend — clasificador de huevos (FastAPI + Keras)

Sirve el modelo del notebook por HTTP para que la app de Expo pueda escanear en "tiempo real"
(foto → predicción → ruta) contra tu instancia EC2.

## 1. Archivos que debes copiar aquí antes de arrancar

Corre el notebook hasta la celda **"Exportación del modelo y la configuración de la banda"**
y copia los dos archivos que genera dentro de `backend/modelo/`:

```
backend/modelo/modelo_huevos_cnn.keras   (o mejor_modelo_huevos.keras)
backend/modelo/config_banda.json
```

Sin esos dos archivos el servidor no arranca (falla rápido y dice cuál falta).

## 2. Subir el backend a la instancia EC2

Con la instancia que mostraste (Ubuntu 24.04, IP pública `34.232.159.175` en tu captura —
verifica la IP actual en la consola de EC2, cambia si reinicias la instancia sin IP elástica):

```bash
# Desde tu máquina, dentro de la carpeta backend/
scp -i tu-llave.pem -r . ubuntu@<IP_PUBLICA>:~/backend
```

O usa la extensión **Remote - SSH** de VS Code para conectarte a `ubuntu@<IP_PUBLICA>` y editar/subir
los archivos directamente ahí.

## 3. Instalar dependencias en la instancia

```bash
ssh -i tu-llave.pem ubuntu@<IP_PUBLICA>
cd ~/backend
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Importante — memoria en `t3.micro` (1 GB RAM):** TensorFlow puede quedarse sin memoria al
instalar o al cargar el modelo en una instancia tan pequeña. Si `pip install` o el arranque del
servidor mueren sin error claro, agrega swap antes de continuar:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

## 4. Abrir el puerto en el Security Group

En la consola EC2 → tu instancia ("bike") → pestaña **Seguridad** → grupo de seguridad →
**Editar reglas de entrada** → agregar regla: `TCP`, puerto `8000`, origen `0.0.0.0/0` (o solo tu
IP si prefieres restringirlo). Sin esto el celular nunca va a poder llegar a la API aunque el
servidor esté corriendo.

## 5. Arrancar el servidor

Para probar rápido (se detiene si cierras la sesión SSH):

```bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

Para dejarlo corriendo aunque cierres la terminal:

```bash
nohup venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 > servidor.log 2>&1 &
```

(Opcional, más robusto: crear un servicio `systemd` que reinicie el proceso solo si se cae —
pídemelo si lo quieres y te paso el archivo `.service`.)

## 6. Probar que responde

Desde tu propio computador (no desde la instancia):

```bash
curl http://<IP_PUBLICA>:8000/salud
```

Debe devolver algo como:

```json
{"status": "ok", "clases": ["Crack", "Fertile", "Infertile"], "img_size": 128, ...}
```

Si responde eso, ya puedes apuntar la app de Expo a `http://<IP_PUBLICA>:8000`.

## Endpoint principal

`POST /predecir`

```json
{
  "imagen": "data:image/jpeg;base64,....",
  "incluir_gradcam": false
}
```

Respuesta:

```json
{
  "clase": "Crack",
  "probabilidades": {"Crack": 0.91, "Fertile": 0.05, "Infertile": 0.04},
  "ruta": "Descarte",
  "latencia_ms": 42.3,
  "gradcam": null
}
```

`incluir_gradcam: true` agrega el mapa de calor superpuesto (JPEG en base64) — actívalo
solo para una foto puntual, no en cada frame del escaneo continuo, porque es más lento.
