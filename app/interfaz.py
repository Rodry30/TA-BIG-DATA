import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
import joblib

# Diccionarios para mostrar opciones legibles y mapear a valores
sexo_dict = {"Hombre": 1, "Mujer": 2}
nivel_educativo_dict = {
    "Sin nivel": 1, "Educación Inicial": 2, "Primaria incompleta": 3, "Primaria completa": 4,
    "Secundaria incompleta": 5, "Secundaria completa": 6, "Básica especial": 7,
    "Superior no universitaria incompleta": 8, "Superior no universitaria completa": 9,
    "Superior universitaria incompleta": 10, "Superior universitaria completa": 11,
    "Maestría/Doctorado": 12
}
lengua_dict = {
    "Quechua": 1, "Aimara": 2, "Ashaninka": 3, "Awajún/Aguaruna": 4, "Shipibo - Konibo": 5,
    "Shawi / Chayahuita": 6, "Matigenka / Machiguenga": 7, "Achuar": 8, "Otra lengua nativa": 9,
    "Castellano": 10, "Portugués": 11, "Otra lengua extranjera": 12, "NO ESCUCHA/NO HABLA": 13,
    "LENGUA DE SEÑAS PERUANAS": 14
}
etnia_dict = {
    "Quechua": 1, "Aimara": 2, "Nativo o Indígena de la Amazonía": 3,
    "Otro Pueblo indígena originario": 4, "Afrodescendiente": 5, "Blanco": 6,
    "Mestizo": 7, "Otro": 8, "NO SABE/NO RESPONDE": 9
}
bin_dict = {"Si": 1, "No": 2}

fields = [
    ("edad", "Edad", "entry", None),
    ("sexo", "Sexo", "combo", list(sexo_dict.keys())),
    ("nivel_educativo_cod", "Nivel educativo", "combo", list(nivel_educativo_dict.keys())),
    ("lengua", "Lengua materna", "combo", list(lengua_dict.keys())),
    ("Etnia", "Etnia", "combo", list(etnia_dict.keys())),
    ("Concentracion", "Limitación para concentrarse", "combo", list(bin_dict.keys())),
    ("Socializar", "Limitación para socializar", "combo", list(bin_dict.keys())),
    ("Experiencia", "¿Ha trabajado antes?", "combo", list(bin_dict.keys())),
    ("Movilidad", "Limitación para moverse", "combo", list(bin_dict.keys())),
]

def cargar_modelo_joblib():
    # Detectar entorno y ruta del modelo joblib
    if os.path.exists("/opt/spark-data"):
        model_path = "/opt/spark-data/modelos/rf_model.joblib"
    else:
        model_path = os.path.join("modelos", "rf_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    return joblib.load(model_path)

def predecir(valores):
    # Convertir a DataFrame Pandas y luego a Spark
    data = {}
    for k, v in valores.items():
        if k == "edad":
            data[k] = [int(v)]
        elif k == "sexo":
            data[k] = [sexo_dict[v]]
        elif k == "nivel_educativo_cod":
            data[k] = [nivel_educativo_dict[v]]
        elif k == "lengua":
            data[k] = [lengua_dict[v]]
        elif k == "Etnia":
            data[k] = [etnia_dict[v]]
        elif k in ["Concentracion", "Socializar", "Experiencia", "Movilidad"]:
            data[k] = [bin_dict[v]]
        else:
            data[k] = [v]
    df_pandas = pd.DataFrame(data)
    # Cargar modelo sklearn (joblib)
    model = cargar_modelo_joblib()
    # Predecir
    try:
        pred = model.predict(df_pandas)[0]
    except Exception as e:
        raise RuntimeError(f"Error al predecir con el modelo: {e}")
    # Intentar obtener probabilidad si existe
    prob = None
    if hasattr(model, 'predict_proba'):
        try:
            prob = model.predict_proba(df_pandas)[0]
        except Exception:
            prob = None

    return pred, prob

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Predicción de Condición de Ocupación (Random Forest)")
        self.geometry("600x600")
        self.inputs = {}
        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self)
        frm.pack(padx=20, pady=20, fill="both", expand=True)
        for i, (key, label, typ, opts) in enumerate(fields):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=5)
            if typ == "entry":
                ent = ttk.Entry(frm)
                ent.grid(row=i, column=1, sticky="ew", pady=5)
                self.inputs[key] = ent
            elif typ == "combo":
                cmb = ttk.Combobox(frm, values=opts, state="readonly")
                cmb.grid(row=i, column=1, sticky="ew", pady=5)
                cmb.current(0)
                self.inputs[key] = cmb
        frm.columnconfigure(1, weight=1)
        btn = ttk.Button(frm, text="Predecir", command=self.on_predict)
        btn.grid(row=len(fields), column=0, columnspan=2, pady=20)
        self.result_lbl = ttk.Label(frm, text="")
        self.result_lbl.grid(row=len(fields)+1, column=0, columnspan=2, pady=10)

    def on_predict(self):
        valores = {k: w.get() for k, w in self.inputs.items()}
        try:
            resultado, prob = predecir(valores)
            if prob is None:
                self.result_lbl.config(text=f"Predicción: {resultado}")
            else:
                # Mostrar probabilidad de la clase predicha
                # Si las clases son strings, obtener índice de la predicción
                try:
                    # Si el modelo es un pipeline, obtener clases_ del estimador final
                    classes = model = None
                except Exception:
                    classes = None
                self.result_lbl.config(text=f"Predicción: {resultado} (probabilidad aproximada)")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo predecir: {e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()