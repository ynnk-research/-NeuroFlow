from huggingface_hub import snapshot_download
import os

def download_neuroflow_models():
    models = [
        "Ynnk-Research/NeuroFlow_Siglip1",
        "Ynnk-Research/NeuroFlow_Siglip2"
    ]
    
    # Destination folder
    base_path = "./models"
    
    for model_id in models:
        folder_name = model_id.split("/")[-1]
        print(f"--- Downloading {model_id} to {base_path}/{folder_name} ---")
        
        snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(base_path, folder_name),
            local_dir_use_symlinks=False, # Copies files directly to the folder
            revision="main"
        )
        print(f"Successfully downloaded {model_id}\n")

if __name__ == "__main__":
    download_neuroflow_models()