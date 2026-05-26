import argparse
import os
import shutil
import logging
import mlflow
import optuna
from pathlib import Path # <-- AGREGADO: Necesario para guardar el modelo final
from dotenv import load_dotenv
from ultralytics import YOLO, settings # <-- AGREGADO: Importamos settings de Ultralytics

# Silenciar logs excesivos de la librería para mantener la terminal limpia
logging.getLogger("ultralytics").setLevel(logging.INFO)
# Cargar variables de entorno desde el archivo .env
load_dotenv()

def objective(trial, data_yaml):
    """
    Función objetivo para Optuna. 
    Prueba una combinación de hiperparámetros en un entrenamiento corto.
    """
    # Definimos el espacio de búsqueda bayesiano
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.7, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    # Cargamos el modelo base (YOLOv8 nano para velocidad de optimización)
    model = YOLO("yolov8n.pt")
    
    # Abrimos un run anidado (nested) en MLflow para rastrear este intento específico
    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
        # Entrenamiento corto solo para evaluar la combinación de parámetros
        model.train(
            data=data_yaml,
            epochs=3,  # Pocas épocas para cuidar el presupuesto de GPU
            imgsz=640,
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            device=0,  # <-- CAMBIADO: Activamos la GPU 0 de RunPod
            plots=False,
            save=False,
            exist_ok=True
        )
        
        # Validamos para extraer las métricas del trial
        metrics = model.val()
        map50_95 = metrics.box.map  # mAP @ 0.5:0.95
        
        # Registramos los parámetros y la métrica objetivo en MLflow
        mlflow.log_params(trial.params)
        mlflow.log_metric("trial_mAP50_95", map50_95)
        
    return map50_95

def main():
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento E2E con YOLO, Optuna y MLflow")
    parser.add_argument("--data_yaml", type=str, required=True, help="Ruta al archivo dataset.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas para el entrenamiento final")
    parser.add_argument("--n_trials", type=int, default=3, help="Número de ejecuciones de optimización")
    parser.add_argument("--model_output", type=str, default="models/best.pt", help="Destino final del modelo entrenado")
    
    args = parser.parse_args()
    
    # <-- AGREGADO: Apagamos el autolog de YOLO para evitar el conflicto INVALID_PARAMETER_VALUE
    settings.update({"mlflow": False})
    
    # Inicializamos el experimento en MLflow
    mlflow.set_experiment("drone-detection-yolo")
    
    # Iniciamos el Run Principal (Padre)
    with mlflow.start_run(run_name="yolo_e2e_pipeline") as parent_run:
        print(f"--- Iniciando Optimización Bayesiana ({args.n_trials} trials) ---")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, args.data_yaml), n_trials=args.n_trials)
        
        best_params = study.best_params
        print(f"Mejores parámetros encontrados: {best_params}")
        
        # Registrar los ganadores en el Run Padre
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        
        print("\n--- Iniciando Entrenamiento Final con Parámetros Ganadores ---")
        final_model = YOLO("yolov8n.pt")
        
        # El entrenamiento real e intensivo
        final_model.train(
            data=args.data_yaml,
            epochs=args.epochs,
            imgsz=640,
            lr0=best_params["lr0"],
            momentum=best_params["momentum"],
            weight_decay=best_params["weight_decay"],
            device=0, # <-- CAMBIADO: Aseguramos el uso de GPU en el entrenamiento final
            project="models",
            name="yolo_drone_production",
            exist_ok=True
        )
        
        # Mapeo y persistencia del artefacto de salida hacia la ruta que controlará DVC
        yolo_internal_best = final_model.trainer.best if hasattr(final_model, 'trainer') else None
        
        # Fallback por si usamos otra versión de Ultralytics
        if not yolo_internal_best or not os.path.exists(yolo_internal_best):
             yolo_internal_best = os.path.join("models", "yolo_drone_production", "weights", "best.pt")

        if yolo_internal_best and os.path.exists(yolo_internal_best):
            output_path = Path(args.model_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(yolo_internal_best, args.model_output)
            
            # Guardamos el modelo como un artefacto oficial
            mlflow.log_artifact(args.model_output, artifact_path="production_model")
            print(f"¡Pipeline finalizado! Modelo respaldado en: {args.model_output}")
        else:
            print(f"Error: No se encontró el archivo de pesos. Se buscó en: {yolo_internal_best}")
            raise FileNotFoundError("YOLO no generó best.pt")

if __name__ == "__main__":
    main()