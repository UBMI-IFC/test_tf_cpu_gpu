# Algunas notas importantes.

## listar GPUs con tensor flow

Python
```python
import tensorflow as tf

tf.config.list_physical_devices('GPU')
```

## Algunas GPUs 'viejas' no salen enlistadas por tener menos nucleos de procesamiento

Ejemplo: NVIDIA GeForce GT 1030

1. primero es necesario cambiar una variable de sistema para acepataral (default=8)

```bash
 TF_MIN_GPU_MULTIPROCESSOR_COUNT=2
 export TF_MIN_GPU_MULTIPROCESSOR_COUNT
```

2. Listo :P Ya se puede usar la tarjeta con Tensorflow


## Revisar el nombre de las tarjetas con tensorflow

```python
for x in tf.config.list_physical_devices('GPU'):
    print(tf.config.experimental.get_device_details(x))
```


