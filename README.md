# tensorflowVSpytorch

vamos a construir una red neuronal simple para clasificar dígitos del dataset MNIST (imágenes 28×28 de dígitos 0–9).

👉 Con Keras/TensorFlow el flujo es muy alto nivel y directo.

👉 En PyTorch se nota que tienes más control explícito sobre el entrenamiento (ciclo for, loss.backward(), optimizer.step()), lo que lo hace muy flexible para investigación.

✅ Comparación rápida:

- Keras/TensorFlow: pocas líneas → ideal para prototipado y producción rápida.

- PyTorch: más explícito → ideal para investigación y arquitecturas personalizadas.

🔹 Flujo general de un proyecto de ML/DL (con ejemplo en el script de este repo)

Un modelo de deep learning sigue cuatro pasos básicos:

Datos: cargar, limpiar y preparar los datos (en este caso imágenes 28×28 píxeles).

Modelo: definir la red neuronal (capas, conexiones, activaciones).

Entrenamiento: mostrarle los datos al modelo muchas veces (épocas), calcular errores y ajustar los pesos.

Evaluación: probar el modelo en datos nuevos (que nunca vio durante el entrenamiento) para medir su precisión.

### Optimización de parámetros
Para mejorar el modelo hay varios métodos en donde se pueden optimizar hiperparámetros. Los métodos clásicos son:

- Búsqueda exhaustiva
- Búsqueda aleatória
- Búsqueda Bayesiana
- Búsqueda orientada por metaheurísticas.
