import argparse
from huggingface_hub import HfApi

def deploy(token):
    api = HfApi(token=token)
    
    print("Uploading repository to Hugging Face...")
    print("This will correctly auto-handle files larger than 10MB without Git LFS errors.")
    
    api.upload_folder(
        folder_path=".",
        repo_id="namansudo/Solarpower",
        repo_type="space",
        ignore_patterns=[
            ".git/*",
            ".git",
            "venv/*",
            "venv",
            "__pycache__/*",
            ".DS_Store",
            "Presentation file/*",
            ".env",
            "deploy_to_hf.py"
        ]
    )
    print("✅ Deployment successful! Check your Space at https://huggingface.co/spaces/namansudo/Solarpower")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Your Hugging Face write token")
    args = parser.parse_args()
    deploy(args.token)
