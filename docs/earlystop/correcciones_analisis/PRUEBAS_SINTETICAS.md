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

Código bajo prueba: `cils_v3_shards.py`, SHA-256 de la versión sintácticamente validada `9a3ac9474bf2da2c35981e0087784e07611ff004ccee4f5ae6e33d7240b9f713`. Ejecutor Slurm propuesto `cils_v3_pilot.sbatch`, SHA-256 `f6e25d5dc387c3f4a5153774f2f1078dcdd8a305ef1b9c78ca813bf534d8e97f`. No se ha enviado el piloto, por tanto no existen todavía su duración, memoria o logs reales.
