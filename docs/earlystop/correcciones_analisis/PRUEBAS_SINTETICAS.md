# EARLYSTOP — pruebas sintéticas locales del checkpoint v1 (13/11/2026)

Pruebas ejecutadas en un entorno local aislado (**NO SCAYLE**) sobre 5 sistemas artificiales, con 3 fragmentos de 2/2/1 sistemas, ambas políticas `include/exclude` y un ejecutor ficticio de objetivos. Estas pruebas **no demuestran validez científica de targets ATLAS**.

**Resultado:** 11/11 comprobaciones PASS:
- Plan exacto 5 sistemas / 3 fragmentos.
- Interrupción simulada antes de escribir resultados.
- Preservación inalterada del fragmento completado antes del fallo.
- Ausencia de checkpoint de un fragmento fallido.
- Reanudación sin recalcular el fragmento ya válido.
- Consolidación bloqueada por fragmentos ausentes.
- Validación detecta fila duplicada.
- Validación detecta fila faltante.
- Verificación detecta alteración del SHA-256 de un fragmento previamente válido.
- Consolidación completa: 90 filas por política.
- Verificación y reutilización de una consolidación completada.

Entorno local de prueba: CPython 3.13.5. **El despliegue en SCAYLE debe repetir `selftest` con Python 3.11.5 antes de enviar el piloto.** Una prueba exitosa local no equivale a haber ejecutado fragmentos ATLAS.

Archivo de resultados: [`PRUEBAS_SINTETICAS.json`](PRUEBAS_SINTETICAS.json).

Versión del prototipo local ejecutado: SHA-256 `9a3ac9474bf2da2c35981e0087784e07611ff004ccee4f5ae6e33d7240b9f713`. El ejecutor publicado en GitHub está en [`cils_v3_shards.py`](cils_v3_shards.py), Git blob SHA-1 `4eb36e04a06d2292415235c342bdca70de655ab8`; es una copia de código cuyo contenido debe pasar su propio `selftest` **en SCAYLE antes del primer sbatch** (no se declara igualdad byte a byte con el prototipo local). El `.sbatch` publicado está en [`cils_v3_pilot.sbatch`](cils_v3_pilot.sbatch), Git blob SHA-1 `1f8d5ff8c6a18d19fc38467ab47cd9bcf26b6488`. No se ha enviado todavía el piloto; su duración, memoria y logs no están acreditados.
