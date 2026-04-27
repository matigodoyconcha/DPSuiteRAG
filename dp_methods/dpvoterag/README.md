# DPVoteRAG

Implementacion del metodo DPVoteRAG basado en Koga et al. (arXiv:2412.04697v3).

## Idea General

DPVoteRAG usa votacion entre `m` votantes para cada paso de generacion token a token.
Cada votante observa un subconjunto de `k` documentos recuperados y propone un siguiente token.
Luego se agrega un histograma de votos y se selecciona el token final con un mecanismo
diferencialmente privado de dominio limitado (LimitedDomain).

## Parametros

- `m`: numero de voters. Valores altos tienden a mejorar robustez del voto, pero aumentan costo.
- `k`: documentos por voter. El paper usa principalmente `k=1`.
- `epsilon_token`: presupuesto de privacidad por token. Mas alto suele mejorar utilidad y bajar ruido.
- `delta_token`: delta por token.
- `epsilon_total`: presupuesto total por respuesta.
- `delta_total`: delta total por respuesta.
- `k_bar`: tamano del dominio reducido en LimitedDomain. Si no se fija, se usa `m`.
- `random_seed`: controla particionado aleatorio y ruido para reproducibilidad.

## Costo Computacional

Cada query requiere aproximadamente `m` inferencias del modelo por token generado.
Por eso DPVoteRAG puede ser mucho mas costoso que RAG vanilla, especialmente con `m` grande
y respuestas largas.

## Nota sobre LimitedDomain

Esta implementacion usa una version simplificada de LimitedDomain:

- Selecciona los `k_bar` tokens mas votados.
- Anade ruido Gumbel con escala `1/epsilon_token`.
- Elige el token con mayor conteo ruidoso.

La parte completa de calibracion de umbrales y tratamiento formal fino de `delta` del mecanismo
original de Durfee & Rogers (NeurIPS 2019) no esta implementada en su forma completa aqui.

## DPSparseVoteRAG (SVT)

DPSparseVoteRAG extiende DPVoteRAG con una etapa previa tipo Sparse Vector Technique
(AboveThreshold) en cada paso de tiempo:

1. Genera un token no privado `y_nonRAG` usando prompt sin contexto.
2. Consulta su conteo en el histograma de voters privados.
3. Si ese conteo (con ruido Laplaciano) no supera el umbral ruidoso `tau_hat`,
   usa LimitedDomain y consume presupuesto privado.
4. Si lo supera, usa `y_nonRAG` directamente y no consume presupuesto privado.

Esto permite reservar presupuesto para pasos donde la señal privada aporta mas.

### Advertencia formal

El mecanismo de ahorro de presupuesto cuando `y_nonRAG` coincide con un token frecuente
del histograma no tiene una justificacion formal completamente cerrada en el paper original.
En esta implementacion, la consulta para AboveThreshold es un conteo con sensibilidad 1,
lo cual es consistente con la configuracion Laplaciana usada. Sin embargo, la interaccion
entre la regeneracion de `tau_hat` y la composicion total del mecanismo merece una revision
formal mas cuidadosa para afirmaciones teoricas estrictas.

## Hyperparametros recomendados (segun tendencias del paper)

| Regimen | m | k | epsilon_token | epsilon_total | tau | tmax_absolute |
| --- | --- | --- | --- | --- | --- | --- |
| Piloto rapido | 10 | 1 | 2 | 10 | m/2 | 120-200 |
| Balanceado | 30 | 1 | 2 | 10-20 | m/2 | 200 |
| Alta utilidad | 40-50 | 1 | 5 | 20-40 | m/2 | 200-256 |

Notas practicas:

- `tau = m/2` funciona como punto inicial robusto.
- Si `tokens_nonprivate` queda cerca de 0, conviene bajar `tau` o subir `epsilon_token`.
- Si hay demasiada longitud pero baja precision, conviene subir `tau` para forzar mas votos privados.
