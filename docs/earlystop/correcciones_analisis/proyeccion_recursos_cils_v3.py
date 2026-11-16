"""Proyecta recursos del resto de fragmentos a partir de las mediciones del piloto 3387582.
Solo aritmética sobre los dos complete.json del fragmento 000 (evidencia_3387582/). No ejecuta ningún análisis.
Supuestos: coste lineal por sistema, igual al de los 10 sistemas del fragmento 000; 757 sistemas en 76 fragmentos (75×10 + 7)."""
import json, math, pathlib
E = pathlib.Path(__file__).resolve().parent / 'evidencia_3387582'
cp = [json.loads((E / f'complete_{p}_000.json').read_text()) for p in ('include', 'exclude')]
t = sum(c['elapsed_seconds'] for c in cp) / len(cp)           # s por tarea (fragmento × política) de 10 sistemas
per_sys = t / 10
rss = max(c['child_maxrss_kib'] for c in cp) / 1024
cpu_ratio = sum(c['child_cpu_seconds'] for c in cp) / sum(c['elapsed_seconds'] for c in cp)
tareas_10 = 2 * 74                                            # fragmentos 1..74 (10 sistemas) × 2 políticas
tareas_7 = 2                                                  # fragmento 75 (7 sistemas) × 2 políticas
rest = tareas_10 * t + tareas_7 * 7 * per_sys
print(f'Tarea (10 sistemas, 1 política): {t:.1f} s = {t/60:.1f} min; {per_sys:.1f} s por sistema y política')
print(f'MaxRSS hijo: {rss:.1f} MiB; CPU/wall = {cpu_ratio:.3f}')
print(f'Resto (150 tareas: 148 de 10 sistemas + 2 de 7): {rest:.0f} s = {rest/3600:.1f} h de CPU (1 núcleo)')
print(f'Total 757 sistemas × 2 políticas: {2*757*per_sys/3600:.1f} h de CPU')
print('\n| Concurrencia (tareas a la vez) | Oleadas | Tiempo de reloj aprox. |\n|---|---|---|')
for n in (5, 10, 15, 25, 50, 75):
    ol = math.ceil(150 / n)
    print(f'| {n} | {ol} | {ol * t / 3600:.1f} h |')
