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

Versión local más reciente ejecutada: SHA-256 `91378cd7ca67a5e01342d4c951793ba945dd06d4621d7aac676cc4b625112899`, con prueba de duplicado por sustitución de una fila sin alterar el recuento. El ejecutor publicado en GitHub está en [`cils_v3_shards.py`](cils_v3_shards.py), Git blob SHA-1 `42a88171b3a2a3144652e294e390476d4264099f`; es una copia de código cuyo contenido debe pasar su propio `selftest` **en SCAYLE antes del primer sbatch** (no se declara igualdad byte a byte con el prototipo local). El `.sbatch` publicado está en [`cils_v3_pilot.sbatch`](cils_v3_pilot.sbatch), Git blob SHA-1 `1f8d5ff8c6a18d19fc38467ab47cd9bcf26b6488`. No se ha enviado todavía el piloto; su duración, memoria y logs no están acreditados.
