import runpod
import os
import time
import subprocess

# Cargamos la API Key desde las variables de entorno de GitHub Actions
runpod.api_key = os.getenv("RUNPOD_API_KEY")

def main():
    print("🚀 Paso 1: Solicitando instancia GPU en RunPod...")
    
    # Lista de GPUs ordenadas por tu preferencia
    gpu_preferences = [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A6000",
        "NVIDIA RTX A4000"
    ]
    
    pod = None
    
    # Mecanismo de Fallback: Intentamos rentar una por una
    for gpu in gpu_preferences:
        print(f"Buscando disponibilidad para: {gpu}...")
        try:
            pod = runpod.create_pod(
                name="MLOps-Training-GPU",
                image_name="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                gpu_type_id=gpu,
                ports="22/tcp"
            )
            print(f"✅ ¡Éxito! Instancia {gpu} reservada.")
            break  # Salimos del ciclo si tuvimos éxito
        except runpod.error.QueryError as e:
            print(f"⚠️ {gpu} sin capacidad en este momento.")
            
    # Si terminamos el ciclo y 'pod' sigue siendo None, ninguna estuvo disponible
    if not pod:
        print("❌ Error crítico: Ninguna de las GPUs solicitadas está disponible en RunPod.")
        exit(1)  # Forzamos un error para que GitHub Actions se detenga y se ponga en rojo
        
    pod_id = pod["id"]
    print(f"⏳ Pod {pod_id} creado. Esperando a que el servidor arranque...")
    
    # Esperamos a que la máquina esté encendida y asigne una IP
    while True:
        pod_info = runpod.get_pod(pod_id)
        if pod_info["desiredStatus"] == "RUNNING" and pod_info.get("runtime"):
            ip = pod_info["runtime"]["ports"][0]["ip"]
            port = pod_info["runtime"]["ports"][0]["externalPort"]
            print(f"✅ Pod listo en IP: {ip} | Puerto: {port}")
            break
        time.sleep(10)
        print("Esperando 10 segundos más...")
    
    # Damos 30 segundos extra para que el servicio SSH interno levante completamente
    time.sleep(30)
    
    print("🛠️ Paso 2: Inyectando comandos por SSH...")
    
    # Construimos el comando SSH. 
    # El archivo 'runpod_key' lo generará GitHub Actions antes de correr este script.
    # El flag -o StrictHostKeyChecking=no evita que SSH nos pregunte "Are you sure?"
    ssh_command = f"ssh -i runpod_key -p {port} -o StrictHostKeyChecking=no root@{ip} "
    
    # La lista de tareas que la GPU debe ejecutar:
    # 1. Clonar el repositorio
    # 2. Instalar dependencias
    # 3. Exportar variables de AWS y MLflow
    # 4. Correr DVC
    remote_script = f"""
    git clone https://github.com/jmvazqueznicolas/drone-detection-mlops.git repo
    cd repo
    pip install -r requirements.txt
    export AWS_ACCESS_KEY_ID={os.getenv('AWS_ACCESS_KEY_ID')}
    export AWS_SECRET_ACCESS_KEY={os.getenv('AWS_SECRET_ACCESS_KEY')}
    export MLFLOW_TRACKING_URI={os.getenv('MLFLOW_TRACKING_URI')}
    export MLFLOW_TRACKING_USERNAME={os.getenv('MLFLOW_TRACKING_USERNAME')}
    export MLFLOW_TRACKING_PASSWORD={os.getenv('MLFLOW_TRACKING_PASSWORD')}
    dvc pull
    dvc repro
    dvc push
    """
    
    # Ejecutamos el comando
    try:
        subprocess.run(ssh_command + f'"{remote_script}"', shell=True, check=True)
        print("🎉 Entrenamiento finalizado con éxito.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error durante el entrenamiento: {e}")
    finally:
        # PASE LO QUE PASE (éxito o error), apagamos la máquina para no gastar saldo.
        print(f"🧹 Paso 3: Destruyendo el Pod {pod_id} para detener facturación...")
        runpod.terminate_pod(pod_id)
        print("✅ Infraestructura limpiada.")

if __name__ == "__main__":
    main()