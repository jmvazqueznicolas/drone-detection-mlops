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
                ports="22/tcp",
                container_disk_in_gb=50,  # <-- Aumentamos el disco principal a 50 GB
                volume_in_gb=50           # <-- Aumentamos el volumen extra a 50 GB
            )
            print(f"✅ ¡Éxito! Instancia {gpu} reservada con 50GB de disco.")
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
        if pod_info.get("desiredStatus") == "RUNNING" and pod_info.get("runtime"):
            # Obtenemos la lista de todos los puertos expuestos
            ports = pod_info["runtime"].get("ports", [])
            
            ip = None
            port = None
            
            # Buscamos específicamente el que mapea al puerto 22 (SSH)
            for p in ports:
                if p.get("privatePort") == 22:
                    ip = p.get("ip")
                    port = p.get("publicPort")
                    break
                    
            if ip and port:
                print(f"✅ Pod listo en IP: {ip} | Puerto (SSH): {port}")
                break
                
        time.sleep(10)
        print("Esperando 10 segundos a que RunPod asigne red...")
    
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
    # La lista de tareas que la GPU debe ejecutar
    # La lista de tareas que la GPU debe ejecutar
    # La lista de tareas que la GPU debe ejecutar
    remote_script = f"""
    set -e  # Freno de emergencia
    
    echo "1. Clonando repositorio..."
    git clone https://x-access-token:{os.getenv('GITHUB_TOKEN')}@github.com/jmvazqueznicolas/drone-detection-mlops.git repo
    cd repo
    
    echo "2. Preparando entorno virtual..."
    python -m venv venv
    source venv/bin/activate
    
    echo "3. Instalando dependencias..."
    pip install --upgrade pip
    
    # 1. Instalamos explícitamente PyTorch con soporte CUDA desde el repositorio oficial
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
    
    # 2. Instalamos el resto de tus dependencias
    pip install --no-cache-dir -r requirements.txt
    
    echo "4. Configurando credenciales..."
    # Usamos echo para evitar el bug de los espacios de EOF
    mkdir -p ~/.aws
    echo "[default]" > ~/.aws/credentials
    echo "aws_access_key_id = {os.getenv('AWS_ACCESS_KEY_ID')}" >> ~/.aws/credentials
    echo "aws_secret_access_key = {os.getenv('AWS_SECRET_ACCESS_KEY')}" >> ~/.aws/credentials
    echo "region = us-east-1" >> ~/.aws/credentials
    
    export MLFLOW_TRACKING_URI="{os.getenv('MLFLOW_TRACKING_URI')}"
    export MLFLOW_TRACKING_USERNAME="{os.getenv('MLFLOW_TRACKING_USERNAME')}"
    export MLFLOW_TRACKING_PASSWORD="{os.getenv('MLFLOW_TRACKING_PASSWORD')}"
    
    echo "5. Ejecutando pipeline MLOps..."
    # Agregamos '|| true' para que DVC descargue los datos y no aborte si falta el modelo viejo
    dvc pull || true
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