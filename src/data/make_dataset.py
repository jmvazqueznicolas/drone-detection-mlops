import argparse
import os
import shutil
import random
import yaml
from pathlib import Path

def setup_yolo_directories(processed_dir):
    """Crea la estructura de carpetas que YOLO requiere, incluyendo test."""
    subdirs = [
        'images/train', 'images/val', 'images/test', 
        'labels/train', 'labels/val', 'labels/test'
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(processed_dir, subdir), exist_ok=True)

def split_and_copy_data(raw_dir, processed_dir, split_ratio, seed=42):
    """
    Divide los datos en train, val y test.
    split_ratio es una tupla ej. (0.7, 0.2, 0.1)
    """
    random.seed(seed)
    
    raw_images_dir = Path(raw_dir) / 'images'
    raw_labels_dir = Path(raw_dir) / 'labels'
    
    # Obtenemos todas las imágenes válidas
    images = [f for f in os.listdir(raw_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)
    
    total_images = len(images)
    train_end = int(total_images * split_ratio[0])
    val_end = train_end + int(total_images * split_ratio[1])
    
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    def copy_files(file_list, split_name):
        for img_name in file_list:
            src_img = raw_images_dir / img_name
            dst_img = Path(processed_dir) / 'images' / split_name / img_name
            shutil.copy(src_img, dst_img)
            
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            src_txt = raw_labels_dir / txt_name
            dst_txt = Path(processed_dir) / 'labels' / split_name / txt_name
            
            if src_txt.exists():
                shutil.copy(src_txt, dst_txt)

    print(f"Copiando {len(train_images)} imágenes a train...")
    copy_files(train_images, 'train')
    
    print(f"Copiando {len(val_images)} imágenes a val...")
    copy_files(val_images, 'val')
    
    print(f"Copiando {len(test_images)} imágenes a test...")
    copy_files(test_images, 'test')

def create_yaml_config(processed_dir, classes):
    """Genera el dataset.yaml para Ultralytics, apuntando a train, val y test."""
    data = {
        # Esta es la ruta relativa desde la raíz del proyecto
        'path': 'data/processed/drones_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    yaml_path = os.path.join(processed_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Archivo {yaml_path} generado con éxito.")

def main():
    parser = argparse.ArgumentParser(description="Preprocesa datos raw a formato YOLO")
    parser.add_argument("--raw_data", type=str, required=True, help="Ruta de entrada (data/raw)")
    parser.add_argument("--processed_data", type=str, required=True, help="Ruta de salida (data/processed)")
    # Permitimos ingresar las proporciones desde la terminal
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.2, 0.1], help="Proporciones para train val test")
    parser.add_argument("--classes", nargs='+', default=["drone"], help="Nombres de las clases")
    
    args = parser.parse_args()
    
    # Validar que el split sume 1
    if sum(args.split) != 1.0:
        raise ValueError("El split debe sumar exactamente 1.0 (ej. 0.7 0.2 0.1)")
        
    print(f"Iniciando preprocesamiento: {args.raw_data} -> {args.processed_data}")
    setup_yolo_directories(args.processed_data)
    split_and_copy_data(args.raw_data, args.processed_data, args.split)
    create_yaml_config(args.processed_data, args.classes)
    print("Preprocesamiento finalizado.")

if __name__ == "__main__":
    main()