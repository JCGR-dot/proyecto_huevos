# Escáner de huevos en tiempo real — CNN + Expo + EC2

Este proyecto conecta el modelo entrenado en el notebook (`Trabajo_Final_CNN_Huevos_Banda.ipynb`)
con una app móvil que escanea huevos en vivo con la cámara del celular.

```
backend/       API FastAPI que carga el modelo (.keras) y expone POST /predecir
mobile-app/    App de Expo (React Native) que usa la cámara y consume esa API
```

## Orden de trabajo recomendado

1. Corre el notebook hasta exportar `modelo_huevos_cnn.keras` y `config_banda.json`.
2. Sigue `backend/README.md` para subir y arrancar la API en tu instancia EC2.
3. Sigue `mobile-app-nuevo/README.md` para correr la app con Expo Go en tu celular, apuntándola a la
   IP pública de esa instancia.

La política de decisión (umbral de `Crack`, confianza mínima, ruta por clase) y el Grad-CAM son
exactamente los que ya validaste en el notebook — el backend solo los reutiliza sobre la foto que
manda el celular en vez de sobre las imágenes de test.
