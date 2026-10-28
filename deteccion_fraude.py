"""
Detección de fraude en transacciones con tarjeta de crédito mediante una red neuronal profunda.

Electiva II – Inteligencia Artificial Avanzada · Unidad 1, Actividad 2
Programa de Ingeniería de Software, Tecnológica del Oriente

Implementa exactamente el diseño descrito en el informe y la presentación:
  - Conjunto de datos: Credit Card Fraud Detection (ULB, Kaggle) — creditcard.csv
  - Preprocesamiento: eliminación de duplicados; escalado de Time (estándar) y Amount (robusto),
    ajustado SOLO con el conjunto de entrenamiento (sin fuga de datos).
  - Partición estratificada 70 % / 15 % / 15 % con semilla fija (42).
  - Red densa (MLP): 30 → Densa 64 ReLU → BatchNorm → Dropout 0,3 → Densa 32 ReLU → Dropout 0,3 → 1 sigmoide.
  - Pérdida: Binary Focal Loss (γ = 2, α = 0,25).
  - Optimizador: AdamW (lr = 1e-3, weight_decay = 1e-4) + ReduceLROnPlateau + detención temprana sobre PR-AUC de validación.
  - Umbral de decisión calibrado en validación (máximo F1).
  - Línea base: regresión logística con los mismos datos y el mismo procedimiento de umbral.
  - Métricas en prueba: sensibilidad, precisión, F1, PR-AUC, ROC-AUC y matriz de confusión.
  - Ética: tasa de falsos positivos por rangos de monto y franja horaria; explicación con SHAP (opcional).

Uso:
  1. Descargar creditcard.csv desde https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  2. pip install torch scikit-learn pandas matplotlib shap
  3. python deteccion_fraude.py --data creditcard.csv
     (en Kaggle: --data /kaggle/input/creditcardfraud/creditcard.csv)

Salidas (carpeta resultados/): resultados.json, tabla_resultados.csv, equidad.csv y las figuras
matriz_confusion.png, curva_pr.png, curvas_entrenamiento.png y shap_importancia.png.

Versión 2: añade --experimentos (ablaciones y robustez por semilla) que genera experimentos_resumen.csv.
El flujo principal y la configuración base no cambian respecto a deteccion_fraude.py.
"""

import argparse
import copy
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # permite generar las figuras sin pantalla (Colab, Kaggle, servidor)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Configuración (coincide con la Tabla 3 del informe)
# ---------------------------------------------------------------------------
CONFIG = {
    "semilla": 42,
    "tasa_aprendizaje": 1e-3,
    "decaimiento_pesos": 1e-4,
    "tamano_lote": 2048,
    "epocas_max": 100,
    "paciencia_detencion": 10,
    "paciencia_lr": 5,
    "factor_lr": 0.5,
    "dropout": 0.3,
    "focal_gamma": 2.0,
    "focal_alpha": 0.25,
}


def fijar_semillas(semilla: int) -> None:
    """Fija todas las fuentes de aleatoriedad para que los resultados sean reproducibles."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 1. Datos: carga, limpieza, partición y escalado
# ---------------------------------------------------------------------------
def cargar_y_preparar(ruta_csv: str, semilla: int):
    df = pd.read_csv(ruta_csv)
    print(f"Transacciones cargadas: {len(df):,} | fraudes: {int(df['Class'].sum()):,} "
          f"({df['Class'].mean() * 100:.3f} %)")

    duplicados = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicados eliminados: {duplicados:,} | quedan {len(df):,} transacciones")

    columnas = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[columnas].copy()
    y = df["Class"].values.astype(np.float32)

    # Partición estratificada 70 / 15 / 15: primero 70 % vs 30 %, luego el 30 % en dos mitades.
    X_ent, X_tmp, y_ent, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=semilla)
    X_val, X_pru, y_val, y_pru = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=semilla)

    # Se guardan los valores originales de monto y hora para el análisis de equidad.
    crudos_prueba = X_pru[["Time", "Amount"]].copy()

    # Escalado: los escaladores se AJUSTAN solo con entrenamiento (prevención de fuga de datos).
    esc_tiempo = StandardScaler().fit(X_ent[["Time"]])
    esc_monto = RobustScaler().fit(X_ent[["Amount"]])  # robusto: el monto es muy asimétrico
    for parte in (X_ent, X_val, X_pru):
        parte["Time"] = esc_tiempo.transform(parte[["Time"]]).ravel()
        parte["Amount"] = esc_monto.transform(parte[["Amount"]]).ravel()

    for nombre, etiquetas in (("Entrenamiento", y_ent), ("Validación", y_val), ("Prueba", y_pru)):
        print(f"  {nombre:<13} {len(etiquetas):>8,} transacciones | {int(etiquetas.sum()):>4} fraudes")

    return (X_ent.values.astype(np.float32), y_ent,
            X_val.values.astype(np.float32), y_val,
            X_pru.values.astype(np.float32), y_pru,
            crudos_prueba, columnas)


# ---------------------------------------------------------------------------
# 2. Modelo: red densa (MLP) — Tabla 2 del informe
# ---------------------------------------------------------------------------
class RedFraude(nn.Module):
    def __init__(self, n_entradas: int = 30, dropout: float = 0.3):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_entradas, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),  # la sigmoide se aplica en la pérdida y en predecir()
        )

    def forward(self, x):
        return self.red(x).squeeze(1)  # logits


class BinaryFocalLoss(nn.Module):
    """Focal Loss binaria (Lin et al., 2017): FL(p_t) = -α_t (1 - p_t)^γ log(p_t)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, objetivos):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, objetivos, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * objetivos + (1 - p) * (1 - objetivos)
        alpha_t = self.alpha * objetivos + (1 - self.alpha) * (1 - objetivos)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


@torch.no_grad()
def predecir(modelo, X, dispositivo, lote: int = 8192) -> np.ndarray:
    modelo.eval()
    salidas = []
    for i in range(0, len(X), lote):
        xb = torch.from_numpy(X[i:i + lote]).to(dispositivo)
        salidas.append(torch.sigmoid(modelo(xb)).cpu().numpy())
    return np.concatenate(salidas)


# ---------------------------------------------------------------------------
# 3. Entrenamiento con AdamW, ReduceLROnPlateau y detención temprana
# ---------------------------------------------------------------------------
def entrenar(X_ent, y_ent, X_val, y_val, cfg, dispositivo, criterio=None):
    generador = torch.Generator().manual_seed(cfg["semilla"])
    cargador = DataLoader(TensorDataset(torch.from_numpy(X_ent), torch.from_numpy(y_ent)),
                          batch_size=cfg["tamano_lote"], shuffle=True, generator=generador)

    modelo = RedFraude(X_ent.shape[1], cfg["dropout"]).to(dispositivo)
    if criterio is None:
        criterio = BinaryFocalLoss(cfg["focal_alpha"], cfg["focal_gamma"])
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=cfg["tasa_aprendizaje"],
                                    weight_decay=cfg["decaimiento_pesos"])
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="max", factor=cfg["factor_lr"], patience=cfg["paciencia_lr"])

    n_param = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_param:,}")

    historial = {"perdida_ent": [], "perdida_val": [], "pr_auc_val": []}
    mejor_pr_auc, mejor_estado, sin_mejora = -1.0, None, 0
    X_val_t = torch.from_numpy(X_val).to(dispositivo)
    y_val_t = torch.from_numpy(y_val).to(dispositivo)

    for epoca in range(1, cfg["epocas_max"] + 1):
        modelo.train()
        perdida_total = 0.0
        for xb, yb in cargador:
            xb, yb = xb.to(dispositivo), yb.to(dispositivo)
            optimizador.zero_grad()
            perdida = criterio(modelo(xb), yb)
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item() * len(xb)

        modelo.eval()
        with torch.no_grad():
            perdida_val = criterio(modelo(X_val_t), y_val_t).item()
        prob_val = predecir(modelo, X_val, dispositivo)
        pr_auc_val = average_precision_score(y_val, prob_val)

        historial["perdida_ent"].append(perdida_total / len(X_ent))
        historial["perdida_val"].append(perdida_val)
        historial["pr_auc_val"].append(pr_auc_val)
        planificador.step(pr_auc_val)

        if pr_auc_val > mejor_pr_auc:
            mejor_pr_auc, mejor_estado, sin_mejora = pr_auc_val, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1

        lr_actual = optimizador.param_groups[0]["lr"]
        print(f"Época {epoca:3d} | pérdida ent {historial['perdida_ent'][-1]:.5f} | "
              f"pérdida val {perdida_val:.5f} | PR-AUC val {pr_auc_val:.4f} | lr {lr_actual:.1e}")

        if sin_mejora >= cfg["paciencia_detencion"]:
            print(f"Detención temprana en la época {epoca}. Mejor PR-AUC de validación: {mejor_pr_auc:.4f}")
            break

    modelo.load_state_dict(mejor_estado)
    return modelo, historial


# ---------------------------------------------------------------------------
# 4. Umbral y métricas
# ---------------------------------------------------------------------------
def umbral_max_f1(y_true, prob) -> float:
    """Selecciona, sobre VALIDACIÓN, el umbral que maximiza la puntuación F1."""
    precision, recall, umbrales = precision_recall_curve(y_true, prob)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(umbrales[int(np.argmax(f1))])


def metricas(y_true, prob, umbral) -> dict:
    y_pred = (prob >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "umbral": round(umbral, 4),
        "sensibilidad": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, prob),
        "roc_auc": roc_auc_score(y_true, prob),
        "exactitud": (vp + vn) / len(y_true),
        "VP": int(vp), "FP": int(fp), "FN": int(fn), "VN": int(vn),
    }


# ---------------------------------------------------------------------------
# 4b. Experimentos: ablaciones y robustez por semilla (mismos datos y partición)
# ---------------------------------------------------------------------------
def ejecutar_experimentos(X_ent, y_ent, X_val, y_val, X_pru, y_pru, cfg, dispositivo,
                          res_lr, historial_ref, res_ref, carpeta):
    """Entrena variantes del MLP y la regresión logística; guarda experimentos_resumen.csv.

    La partición de datos es la misma (semilla 42) en todos los experimentos; solo cambia lo indicado.
    """
    filas = []

    def registrar(nombre, descripcion, semilla, res, historial, segundos):
        filas.append({
            "experimento": nombre,
            "descripcion": descripcion,
            "semilla": semilla,
            "epocas_entrenadas": len(historial["perdida_ent"]) if historial else "",
            "mejor_epoca": int(np.argmax(historial["pr_auc_val"])) + 1 if historial else "",
            **{k: res[k] for k in ("umbral", "sensibilidad", "precision", "f1", "pr_auc", "roc_auc",
                                   "VP", "FP", "FN", "VN")},
            "segundos": round(segundos, 1),
        })

    # Referencia: el MLP + Focal Loss ya entrenado en el flujo principal (no se reentrena).
    registrar("mlp_focal_ref", "MLP + Focal Loss (configuración base, ejecución principal)",
              cfg["semilla"], res_ref, historial_ref, 0.0)
    registrar("logreg_ref", "Regresión logística (línea base, ejecución principal)",
              cfg["semilla"], res_lr, None, 0.0)

    # Regresión logística con pesos de clase balanceados
    print("\n[Experimento] Regresión logística con class_weight='balanced'...")
    t0 = time.time()
    lr_b = LogisticRegression(max_iter=2000, random_state=cfg["semilla"], class_weight="balanced").fit(X_ent, y_ent)
    umbral_b = umbral_max_f1(y_val, lr_b.predict_proba(X_val)[:, 1])
    res_b = metricas(y_pru, lr_b.predict_proba(X_pru)[:, 1], umbral_b)
    registrar("logreg_balanceada", "Regresión logística con class_weight='balanced'",
              cfg["semilla"], res_b, None, time.time() - t0)

    peso_pos = float((1 - y_ent.mean()) / y_ent.mean())
    variantes = [
        ("mlp_bce", "MLP + entropía cruzada binaria sin ponderar", cfg,
         lambda: nn.BCEWithLogitsLoss()),
        ("mlp_bce_pos_weight", f"MLP + entropía cruzada con pos_weight = {peso_pos:.1f} (n_neg/n_pos)", cfg,
         lambda: nn.BCEWithLogitsLoss(pos_weight=torch.tensor(peso_pos, device=dispositivo))),
        ("mlp_focal_sin_dropout", "MLP + Focal Loss sin Dropout (dropout = 0)", {**cfg, "dropout": 0.0},
         lambda: BinaryFocalLoss(cfg["focal_alpha"], cfg["focal_gamma"])),
    ]
    for semilla in (1, 2, 3, 4):
        variantes.append((f"mlp_focal_semilla{semilla}",
                          f"MLP + Focal Loss, semilla de inicialización/barajado = {semilla}",
                          {**cfg, "semilla": semilla},
                          lambda: BinaryFocalLoss(cfg["focal_alpha"], cfg["focal_gamma"])))

    for nombre, descripcion, cfg_exp, fabrica_perdida in variantes:
        print(f"\n[Experimento] {nombre}: {descripcion}")
        t0 = time.time()
        fijar_semillas(cfg_exp["semilla"])
        modelo_exp, hist_exp = entrenar(X_ent, y_ent, X_val, y_val, cfg_exp, dispositivo,
                                        criterio=fabrica_perdida())
        umbral_exp = umbral_max_f1(y_val, predecir(modelo_exp, X_val, dispositivo))
        res_exp = metricas(y_pru, predecir(modelo_exp, X_pru, dispositivo), umbral_exp)
        registrar(nombre, descripcion, cfg_exp["semilla"], res_exp, hist_exp, time.time() - t0)

    resumen = pd.DataFrame(filas)
    resumen.to_csv(os.path.join(carpeta, "experimentos_resumen.csv"), index=False)
    print("\n=== Resumen de experimentos (PRUEBA) ===")
    print(resumen[["experimento", "semilla", "epocas_entrenadas", "umbral", "sensibilidad", "precision",
                   "f1", "pr_auc", "roc_auc", "VP", "FP", "FN"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ---------------------------------------------------------------------------
# 5. Análisis de equidad: tasa de falsos positivos por segmento
# ---------------------------------------------------------------------------
def analisis_equidad(crudos, y_true, y_pred) -> pd.DataFrame:
    df = crudos.copy()
    df["real"], df["pred"] = y_true.astype(int), y_pred
    cortes = df["Amount"].quantile([0, 0.25, 0.5, 0.75, 1]).values
    etiquetas = [f"Q{i + 1}: {cortes[i]:.2f}–{cortes[i + 1]:.2f}" for i in range(4)]
    df["rango_monto"] = pd.qcut(df["Amount"], q=4, labels=etiquetas, duplicates="drop").astype(str)
    hora = (df["Time"] % 86400) // 3600
    df["franja_horaria"] = pd.cut(hora, bins=[-1, 5, 11, 17, 23],
                                  labels=["Madrugada (0-5 h)", "Mañana (6-11 h)", "Tarde (12-17 h)", "Noche (18-23 h)"]).astype(str)
    filas = []
    for variable in ("rango_monto", "franja_horaria"):
        for segmento, g in df.groupby(variable):
            legitimas = g[g["real"] == 0]
            fraudes = g[g["real"] == 1]
            filas.append({
                "variable": variable,
                "segmento": segmento,
                "transacciones": len(g),
                "fraudes": len(fraudes),
                "tasa_falsos_positivos": (legitimas["pred"] == 1).mean() if len(legitimas) else np.nan,
                "sensibilidad": (fraudes["pred"] == 1).mean() if len(fraudes) else np.nan,
            })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 6. Figuras
# ---------------------------------------------------------------------------
def guardar_figuras(y_pru, prob_mlp, prob_lr, umbral, historial, carpeta):
    # Matriz de confusión normalizada por fila (Figura 1 del informe)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_pru, (prob_mlp >= umbral).astype(int), display_labels=["Legítima", "Fraude"],
        normalize="true", values_format=".3f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Matriz de confusión normalizada (prueba)")
    ax.set_xlabel("Predicción"); ax.set_ylabel("Valor real")
    fig.tight_layout(); fig.savefig(os.path.join(carpeta, "matriz_confusion.png"), dpi=200); plt.close(fig)

    # Curva de precisión y exhaustividad (Figura 2 del informe)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for nombre, prob, color in (("MLP + Focal Loss", prob_mlp, "#1B7676"), ("Regresión logística", prob_lr, "#8A96A6")):
        p, r, _ = precision_recall_curve(y_pru, prob)
        ax.plot(r, p, color=color, lw=2, label=f"{nombre} (PR-AUC = {average_precision_score(y_pru, prob):.3f})")
    ax.axhline(y_pru.mean(), ls="--", color="#A83A2C", lw=1, label=f"Azar ({y_pru.mean():.4f})")
    ax.set_xlabel("Sensibilidad (recall)"); ax.set_ylabel("Precisión")
    ax.set_title("Curva de precisión y exhaustividad (prueba)"); ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(carpeta, "curva_pr.png"), dpi=200); plt.close(fig)

    # Curvas de entrenamiento
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    epocas = range(1, len(historial["perdida_ent"]) + 1)
    a1.plot(epocas, historial["perdida_ent"], label="Entrenamiento", color="#14284B")
    a1.plot(epocas, historial["perdida_val"], label="Validación", color="#1B7676")
    a1.set_title("Pérdida (Focal Loss)"); a1.set_xlabel("Época"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(epocas, historial["pr_auc_val"], color="#1B7676")
    a2.set_title("PR-AUC de validación"); a2.set_xlabel("Época"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(carpeta, "curvas_entrenamiento.png"), dpi=200); plt.close(fig)


def explicar_con_shap(modelo, X_ent, X_pru, columnas, carpeta, dispositivo, semilla):
    """Importancia global de variables con SHAP (Lundberg & Lee, 2017). Opcional."""
    try:
        import shap
    except ImportError:
        print("SHAP no está instalado (pip install shap); se omite la explicación.")
        return
    rng = np.random.default_rng(semilla)
    fondo = torch.from_numpy(X_ent[rng.choice(len(X_ent), 500, replace=False)]).to(dispositivo)
    muestra = torch.from_numpy(X_pru[rng.choice(len(X_pru), 1000, replace=False)]).to(dispositivo)
    modelo.eval()
    # SHAP necesita una salida 2D (N, 1): se explica el logit de la red sin aplanar.
    envoltura = nn.Sequential(modelo.red).eval()
    try:
        explicador = shap.GradientExplainer(envoltura, fondo)
        valores = np.asarray(explicador.shap_values(muestra)).reshape(len(muestra), -1)
    except Exception as error:  # SHAP cambia su API entre versiones; no debe detener el programa
        print(f"No fue posible calcular SHAP ({error}); se omite la explicación.")
        return
    plt.figure()
    shap.summary_plot(valores, muestra.cpu().numpy(), feature_names=columnas, plot_type="bar",
                      max_display=15, show=False)
    plt.title("Importancia de variables (SHAP)")
    plt.tight_layout(); plt.savefig(os.path.join(carpeta, "shap_importancia.png"), dpi=200); plt.close()


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Detección de fraude con una red neuronal profunda")
    parser.add_argument("--data", default="creditcard.csv", help="Ruta a creditcard.csv")
    parser.add_argument("--salida", default="resultados", help="Carpeta de resultados")
    parser.add_argument("--sin-shap", action="store_true", help="Omitir el análisis SHAP")
    parser.add_argument("--experimentos", action="store_true",
                        help="Ejecutar ablaciones y semillas adicionales (genera experimentos_resumen.csv)")
    args = parser.parse_args()

    cfg = CONFIG
    fijar_semillas(cfg["semilla"])
    os.makedirs(args.salida, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}")

    X_ent, y_ent, X_val, y_val, X_pru, y_pru, crudos_pru, columnas = cargar_y_preparar(args.data, cfg["semilla"])

    # Referencia: exactitud de un clasificador que siempre dice «legítima»
    print(f"\nExactitud de predecir siempre «legítima» en prueba: {(1 - y_pru.mean()) * 100:.2f} %")

    # --- Línea base: regresión logística ---
    print("\nEntrenando la línea base (regresión logística)...")
    lr = LogisticRegression(max_iter=2000, random_state=cfg["semilla"]).fit(X_ent, y_ent)
    umbral_lr = umbral_max_f1(y_val, lr.predict_proba(X_val)[:, 1])
    prob_lr = lr.predict_proba(X_pru)[:, 1]
    res_lr = metricas(y_pru, prob_lr, umbral_lr)

    # --- Modelo propuesto: MLP con Focal Loss ---
    print("\nEntrenando la red neuronal (MLP + Focal Loss)...")
    modelo, historial = entrenar(X_ent, y_ent, X_val, y_val, cfg, dispositivo)
    umbral = umbral_max_f1(y_val, predecir(modelo, X_val, dispositivo))
    prob_mlp = predecir(modelo, X_pru, dispositivo)
    res_mlp = metricas(y_pru, prob_mlp, umbral)

    # --- Resultados (Tabla 4 del informe) ---
    tabla = pd.DataFrame([{"modelo": "Regresión logística (línea base)", **res_lr},
                          {"modelo": "MLP con Focal Loss (propuesto)", **res_mlp}])
    tabla.to_csv(os.path.join(args.salida, "tabla_resultados.csv"), index=False)
    print("\n=== Resultados sobre el conjunto de PRUEBA ===")
    print(tabla[["modelo", "umbral", "sensibilidad", "precision", "f1", "pr_auc", "roc_auc", "VP", "FP", "FN"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- Equidad ---
    equidad = analisis_equidad(crudos_pru, y_pru, (prob_mlp >= umbral).astype(int))
    equidad.to_csv(os.path.join(args.salida, "equidad.csv"), index=False)
    print("\n=== Equidad: tasa de falsos positivos por segmento (prueba) ===")
    print(equidad.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- Figuras, SHAP y guardado ---
    guardar_figuras(y_pru, prob_mlp, prob_lr, umbral, historial, args.salida)
    if not args.sin_shap:
        explicar_con_shap(modelo, X_ent, X_pru, columnas, args.salida, dispositivo, cfg["semilla"])

    torch.save(modelo.state_dict(), os.path.join(args.salida, "modelo_fraude.pt"))
    with open(os.path.join(args.salida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump({"configuracion": cfg, "linea_base": res_lr, "mlp_focal_loss": res_mlp},
                  f, ensure_ascii=False, indent=2, default=float)
    if args.experimentos:
        ejecutar_experimentos(X_ent, y_ent, X_val, y_val, X_pru, y_pru, cfg, dispositivo,
                              res_lr, historial, res_mlp, args.salida)

    print(f"\nListo. Resultados y figuras guardados en la carpeta «{args.salida}».")


if __name__ == "__main__":
    main()
