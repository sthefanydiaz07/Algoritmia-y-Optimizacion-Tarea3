"""Tarea 3 - Ajuste del parámetro de temperatura (T0) del Recocido Simulado con DOE."""
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

MATRICULA = 2163269
T_MIN, T_MAX = 1.0, 100.0
TAMANOS = [5, 10, 15]
REPLICAS = 10
N_ITER = 3000
ALPHA = 0.05
B_BOOT = 10000

T_FINAL = 0.01

random.seed(MATRICULA)
r = random.random()

FRACCIONES = {"bajo": 0.3 * r, "medio": 0.3 * r + 0.3, "alto": 0.3 * r + 0.6}
NIVELES = {nombre: T_MIN + f * (T_MAX - T_MIN) for nombre, f in FRACCIONES.items()}


def tour_length(points, route) -> float:
    return sum(math.dist(points[route[i]], points[route[(i + 1) % len(route)]])
               for i in range(len(route)))


def two_opt_move(route: list[int], i: int, k: int) -> list[int]:
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def linear_schedule(t: int, n_iter: int, T0: float) -> float:
    return T0 + (T_FINAL - T0) * t / n_iter


def exponential_schedule(t: int, n_iter: int, T0: float) -> float:
    return T0 * (T_FINAL / T0) ** (t / n_iter)


SCHEDULE = exponential_schedule


def simulated_annealing(points, initial, schedule, n_iter, seed, T0):
    n = len(initial)
    rng = random.Random(seed)
    current, best = initial[:], initial[:]
    cur_len = best_len = tour_length(points, current)
    for t in range(1, n_iter + 1):
        T = schedule(t, n_iter, T0)
        i = rng.randint(1, n - 2)
        k = rng.randint(i + 1, n - 1)
        nxt = two_opt_move(current, i, k)
        nxt_len = tour_length(points, nxt)
        delta_E = cur_len - nxt_len
        if delta_E > 0:
            current, cur_len = nxt, nxt_len
        elif rng.random() < math.exp(delta_E / T):
            current, cur_len = nxt, nxt_len
        if cur_len < best_len:
            best, best_len = current[:], cur_len
    return best


def random_tour(n: int, seed: int) -> list[int]:
    route = list(range(n))
    random.Random(seed).shuffle(route)
    return route


def crear_instancias():
    instancias = {}
    for n in TAMANOS:
        rng = random.Random(7 + n)
        instancias[n] = [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
    return instancias


def correr_experimento(instancias) -> pd.DataFrame:
    filas = []
    corrida = 0
    for n in TAMANOS:
        puntos = instancias[n]
        for nombre, T0 in NIVELES.items():
            for rep in range(REPLICAS):
                corrida += 1
                semilla = corrida
                inicio = random_tour(n, seed=100000 + semilla)
                mejor = simulated_annealing(puntos, inicio, SCHEDULE, N_ITER, semilla, T0)
                filas.append({"tamano": n, "nivel": nombre, "temperatura": round(T0, 2),
                              "replica": rep + 1, "costo": tour_length(puntos, mejor)})
    return pd.DataFrame(filas)


def bootstrap_mediana(valores, B=B_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(valores)
    medianas = np.median(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return float(np.median(medianas)), float(np.percentile(medianas, 2.5)), float(np.percentile(medianas, 97.5))


def graficas(df: pd.DataFrame):
    orden = list(NIVELES.values())
    orden = [round(t, 2) for t in orden]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    media_t = df.groupby("temperatura")["costo"].mean().reindex(orden)
    axes[0].plot(range(3), media_t.values, "o-", color="#0072B2", linewidth=2)
    axes[0].set_xticks(range(3), [f"{t}\n({nom})" for t, nom in zip(orden, NIVELES)])
    axes[0].set_xlabel("Temperatura inicial T0")
    axes[0].set_ylabel("Costo promedio")
    axes[0].set_title("Efecto principal: temperatura")
    axes[0].set_ylim(media_t.mean() * 0.95, media_t.mean() * 1.05)
    media_n = df.groupby("tamano")["costo"].mean()
    axes[1].plot(range(len(TAMANOS)), media_n.values, "o-", color="#D55E00", linewidth=2)
    axes[1].set_xticks(range(len(TAMANOS)), [str(n) for n in TAMANOS])
    axes[1].set_xlabel("Tamaño del problema (n)")
    axes[1].set_ylabel("Costo promedio")
    axes[1].set_title("Efecto principal: tamaño")
    for a in axes:
        a.grid(alpha=0.25)
    plt.savefig("efectos_principales.png", dpi=150)

    medias = df.groupby(["tamano", "temperatura"])["costo"].mean().unstack().reindex(columns=orden)
    colores = {5: "#009E73", 10: "#0072B2", 15: "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for n in TAMANOS:
        axes[0].plot(range(3), medias.loc[n].values, "o-", color=colores[n], linewidth=2, label=f"n = {n}")
        axes[1].plot(range(3), (medias.loc[n] / medias.loc[n].mean()).values, "o-",
                     color=colores[n], linewidth=2, label=f"n = {n}")
    for a, tit, yl in ((axes[0], "Interacción (costo crudo)", "Costo promedio"),
                       (axes[1], "Interacción (costo / promedio del tamaño)", "Costo relativo")):
        a.set_xticks(range(3), [f"{t}\n({nom})" for t, nom in zip(orden, NIVELES)])
        a.set_xlabel("Temperatura inicial T0")
        a.set_ylabel(yl)
        a.set_title(tit)
        a.grid(alpha=0.25)
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    plt.savefig("interaccion.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    print(f"r = {r:.4f}")
    print("Niveles de temperatura:")
    for nombre, T0 in NIVELES.items():
        print(f"  {nombre:6s} fracción = {FRACCIONES[nombre]:.4f}  ->  T0 = {T0:.2f}")
    assert min(NIVELES.values()) > T_FINAL, "T_MIN debe ser mayor que T_FINAL"

    instancias = crear_instancias()
    df = correr_experimento(instancias)
    df.to_csv("resultados_doe.csv", index=False)
    print(f"\n{len(df)} corridas guardadas en resultados_doe.csv")

    modelo = ols("costo ~ C(temperatura) * C(tamano)", data=df).fit()
    tabla_anova = sm.stats.anova_lm(modelo, typ=2)
    print("\nANOVA de dos factores:")
    print(tabla_anova)
    tabla_anova.to_csv("anova.csv")

    p_temp = tabla_anova.loc["C(temperatura)", "PR(>F)"]
    p_tam = tabla_anova.loc["C(tamano)", "PR(>F)"]
    p_int = tabla_anova.loc["C(temperatura):C(tamano)", "PR(>F)"]
    for nombre, p in (("temperatura", p_temp), ("tamaño", p_tam), ("interacción", p_int)):
        print(f"  {nombre:12s} p = {p:.4g}  ->  {'SIGNIFICATIVO' if p < ALPHA else 'no significativo'}")

    efecto_sig = (p_temp < ALPHA) or (p_int < ALPHA)
    filas = []
    for (n, T0), grupo in df.groupby(["tamano", "temperatura"]):
        med, lo, hi = bootstrap_mediana(grupo["costo"].values)
        filas.append({"tamano": n, "temperatura": T0, "mediana_boot": med, "ic_inf": lo, "ic_sup": hi,
                      "media": grupo["costo"].mean()})
    boot = pd.DataFrame(filas)
    boot["efecto_temp_o_interaccion_significativo"] = efecto_sig
    boot.to_csv("bootstrap_por_punto.csv", index=False)
    print("\nBootstrap (mediana e IC 95%) por punto de diseño:")
    print(boot.round(2).to_string(index=False))

    boot["med_r"] = boot["mediana_boot"].round(6)
    boot["media_r"] = boot["media"].round(6)
    boot["ic_r"] = boot["ic_sup"].round(6)
    ordenado = boot.sort_values(["tamano", "med_r", "media_r", "ic_r"])
    mejores = ordenado.groupby("tamano").head(1).reset_index(drop=True)
    mejores["empate_tecnico"] = [
        ((boot["tamano"] == f.tamano) & (boot["med_r"] == f.med_r) & (boot["media_r"] == f.media_r)).sum() > 1
        for f in mejores.itertuples()]
    mejores = mejores.drop(columns=["med_r", "media_r", "ic_r"])
    mejores.to_csv("resumen_optimos.csv", index=False)
    print("\nTemperatura con mejor desempeño por tamaño:")
    print(mejores[["tamano", "temperatura", "mediana_boot", "ic_inf", "ic_sup", "media", "empate_tecnico"]]
          .round(2).to_string(index=False))

    graficas(df)
    print("\nGráficas guardadas: efectos_principales.png, interaccion.png")
    plt.show()