import argparse
import os
import sys
import datetime
import logging
import hashlib
from pathlib import Path

# Ensure UTF-8 output on Windows terminals (avoids UnicodeEncodeError for -> etc.)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from dotenv import load_dotenv

# Enable logging for Azure ML script operations while keeping HTTP requests clean
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request/response dumping and connection pool warnings
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.ERROR)



try:
    from azure.ai.ml import MLClient, command, Input
    from azure.ai.ml.entities import Data, Environment, BuildContext
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.identity import DefaultAzureCredential
except ImportError:
    MLClient = None
    command = None
    Input = None
    Data = None
    Environment = None
    BuildContext = None
    AssetTypes = None
    InputOutputModes = None
    DefaultAzureCredential = None


# Load Azure connection details (but NOT hyperparameters)
load_dotenv()

AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE_NAME = os.environ.get("AZURE_WORKSPACE_NAME")
AZURE_COMPUTE_NAME = os.environ.get("AZURE_COMPUTE_NAME", "gpu-vm-clean")
AZURE_DATASET_NAME = os.environ.get("AZURE_DATASET_NAME", "car_parts_seg_dataset")
EXPERIMENT_NAME = "car-damage-ml"

def get_ml_client():
    print("  [STEP 1/4] Authenticating with Azure ML Workspace...")
    print(f"             Subscription: {AZURE_SUBSCRIPTION_ID}")
    print(f"             Resource Group: {AZURE_RESOURCE_GROUP}")
    print(f"             Workspace: {AZURE_WORKSPACE_NAME}")
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential, 
            AZURE_SUBSCRIPTION_ID, 
            AZURE_RESOURCE_GROUP, 
            AZURE_WORKSPACE_NAME
        )
        print("  [OK] Azure ML authentication successful!")
        return ml_client
    except Exception as e:
        print(f"[ERROR] Azure authentication failed: {e}")
        sys.exit(1)


def handle_dataset_selection(ml_client, local_dataset_path, auto_upload=False, target_version=None):
    if target_version:
        print(f"[OK] Using specified dataset version: {target_version}")
        return target_version

    print(f"\n--- Checking existing versions for dataset: '{AZURE_DATASET_NAME}' ---")
    
    existing_versions = []
    try:
        data_assets = ml_client.data.list(name=AZURE_DATASET_NAME)
        for asset in data_assets:
            existing_versions.append(asset)
    except Exception as e:
        print(f"[INFO] No existing dataset found or error accessing: {e}")

    try:
        existing_versions.sort(key=lambda x: int(x.version))
    except ValueError:
        existing_versions.sort(key=lambda x: x.version)

    if not existing_versions:
        print("  No existing versions found in Azure ML.")
    else:
        for asset in existing_versions:
            created_at = asset.creation_context.created_at if asset.creation_context else "Unknown"
            print(f"  Version: {asset.version} | Created: {created_at}")

    if auto_upload:
        if existing_versions:
            latest_ver = existing_versions[-1].version
            print(f"  [STEP 2/4] Found existing dataset version '{latest_ver}' on Azure ML. Re-using (Skipping upload!).")
            return latest_ver
        else:
            choice = "2"
            print("  [STEP 2/4] No existing dataset version found on Azure ML. Initializing first upload...")
    else:
        print("\nDo you want to:")
        print("  1) Use an EXISTING version from Azure ML")
        print("  2) UPLOAD the local dataset as a NEW version")
        choice = input("Enter choice [1-2] (default 1 if versions exist else 2): ").strip() or ("1" if existing_versions else "2")
    
    if choice == "1":
        if not existing_versions:
            print("[ERROR] No existing versions to choose from. Exiting.")
            sys.exit(1)
        ver_choice = input(f"Enter the version number to use (default {existing_versions[-1].version}): ").strip() or existing_versions[-1].version
        print(f"[OK] Proceeding with existing dataset version: {ver_choice}")
        return ver_choice
        
    elif choice == "2":
        if not os.path.exists(local_dataset_path):
            print(f"[ERROR] Local dataset path not found: {local_dataset_path}")
            sys.exit(1)
            
        next_version = "1"
        if existing_versions:
            try:
                next_version = str(int(existing_versions[-1].version) + 1)
            except ValueError:
                next_version = "1"
                
        print(f"  [STEP 2/4] Uploading dataset '{local_dataset_path}' to Azure ML as version {next_version}...")
        print("             This uploads files to Azure Storage Blob. Please wait...")
        
        my_data = Data(
            path=local_dataset_path,
            type=AssetTypes.URI_FOLDER,
            description="Car parts dataset",
            name=AZURE_DATASET_NAME,
            version=next_version
        )
        
        ml_client.data.create_or_update(my_data)
        print(f"  [OK] Dataset version {next_version} registered successfully!")
        return next_version
        
    else:
        print("[ERROR] Invalid choice. Exiting.")
        sys.exit(1)

def _compute_docker_hash():
    """SHA-256 of Dockerfile + requirements.txt. Changes -> new env version needed."""
    project_root = Path(".").resolve()
    hasher = hashlib.sha256()
    for fname in ["Dockerfile", "requirements.txt"]:
        fpath = project_root / fname
        if fpath.exists():
            hasher.update(fname.encode())          # include filename in hash
            hasher.update(fpath.read_bytes())
        else:
            print(f"  [WARN] {fname} not found in project root -- hash will be partial.")
    return hasher.hexdigest()[:16]   # 16-char prefix is plenty for a tag


def handle_environment_selection(ml_client, force_rebuild=False):
    env_name = "car-parts-env"
    existing_envs = []
    try:
        for e in ml_client.environments.list(name=env_name):
            existing_envs.append(e)
    except Exception as e:
        logger.info(f"No existing environment list for '{env_name}': {e}")

    try:
        existing_envs.sort(key=lambda x: int(x.version))
    except Exception:
        existing_envs.sort(key=lambda x: str(x.version))

    current_hash = _compute_docker_hash()
    print(f"\n  [STEP 3/4] Docker build context hash: {current_hash}")

    # Check whether the latest registered env was built from the same files
    if existing_envs and not force_rebuild:
        latest = existing_envs[-1]
        latest_ver = latest.version
        stored_hash = (latest.tags or {}).get("docker_hash", None)

        if stored_hash == current_hash:
            print(f"  [STEP 3/4] No changes detected in Dockerfile/requirements.txt.")
            print(f"             Re-using registered Azure ML Environment: '{env_name}:{latest_ver}'")
            print(f"             Skipping Docker build & ACR image push (0s build wait)!")
            return f"{env_name}:{latest_ver}"
        else:
            reason = "force_rebuild requested" if force_rebuild else f"file hash changed ({stored_hash or 'unknown'} -> {current_hash})"
            print(f"  [STEP 3/4] Docker context changed - rebuilding environment ({reason}).")
    elif force_rebuild and existing_envs:
        print(f"  [STEP 3/4] --rebuild_env flag set - forcing new environment build.")
    else:
        print(f"  [STEP 3/4] No existing environment found - building from Dockerfile.")

    next_ver = "1"
    if existing_envs:
        try:
            next_ver = str(int(existing_envs[-1].version) + 1)
        except ValueError:
            next_ver = "1"

    print(f"             Registering and building '{env_name}:{next_ver}'...")
    env_entity = Environment(
        name=env_name,
        version=next_ver,
        description="Environment for YOLO, Mask R-CNN, Fast R-CNN, Mask2Former, SAM2",
        build=BuildContext(path=".", dockerfile_path="Dockerfile"),
        tags={"docker_hash": current_hash},
    )
    registered_env = ml_client.environments.create_or_update(env_entity)
    print(f"  [OK] Azure ML Environment '{env_name}:{next_ver}' created and registered!")
    print(f"       Hash tag stored: docker_hash={current_hash}")
    return registered_env

import shutil
import tempfile

def prepare_clean_code_snapshot():
    """
    Creates an isolated lightweight code snapshot in system temp folder containing ONLY scripts/ and essential config files.
    Completely isolated from local workspace files, git history, or large datasets.
    """
    project_root = Path(".").resolve()
    temp_dir = Path(tempfile.mkdtemp(prefix="azure_code_"))

    for fname in ["Dockerfile", "requirements.txt", "orchestrator.py", ".env"]:
        fpath = project_root / fname
        if fpath.exists():
            shutil.copy2(fpath, temp_dir / fname)

    scripts_src = project_root / "scripts"
    if scripts_src.exists():
        shutil.copytree(
            scripts_src,
            temp_dir / "scripts",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log", "*.excel", "*.xlsx", "job_logs*", "test_logs*", "test_result*"),
            dirs_exist_ok=True
        )

    code_files = list(temp_dir.rglob("*"))
    print(f"  [OK] Clean isolated code snapshot prepared in OS temp ({len(code_files)} items):")
    print(f"       {temp_dir}")
    return str(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Submit training job to Azure ML.")
    # Arguments MUST be provided (no defaults/reading from .env)
    parser.add_argument("--model", required=True, choices=["yolo11m-seg", "maskrcnn", "mask2former", "capacity_check", "all"], help="Model type to train.")
    parser.add_argument("--local_dataset_dir", required=True, help="Path to the local dataset directory.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, required=True, help="Batch size.")
    parser.add_argument("--workers", type=int, required=True, help="Number of dataloader workers.")
    parser.add_argument("--auto_upload", action="store_true", help="Automatically upload dataset without prompting.")
    parser.add_argument("--dataset_version", help="Specific dataset version to use.")
    parser.add_argument("--rebuild_env", action="store_true", help="Force rebuilding a new Azure ML Docker environment version.")
    
    args = parser.parse_args()

    ml_client = get_ml_client()
    
    # 1. Dataset selection / upload
    dataset_version = handle_dataset_selection(ml_client, args.local_dataset_dir, auto_upload=args.auto_upload, target_version=args.dataset_version)

    # 2. Environment selection / reuse (bypasses Docker build on subsequent runs)
    env = handle_environment_selection(ml_client, force_rebuild=args.rebuild_env)

    # 3. Prepare clean code snapshot folder to avoid uploading bloated workspace directories
    code_dir = prepare_clean_code_snapshot()
    
    # 4. Build the command based on the model
    # Copy dataset from downloaded temp path -> ./dataset/ (clean local SSD path)
    # This is the key trick: avoids slow /tmp/dataset_... I/O, matches local training paths
    copy_cmd = "mkdir -p ./dataset && cp -r ${{inputs.dataset}}/. ./dataset/ && echo '[OK] Dataset copied to local SSD.'"

    cmd_export = "python scripts/evaluation/export_excel_report.py --run_dir outputs --output_file model_comparison_report.xlsx"

    if args.model == "capacity_check":
        command_string = f"{copy_cmd} && python scripts/training/capacity_check.py --dataset ./dataset --mode local"
    elif args.model == "all":
        yolo_b = args.batch if (args.batch > 0 and args.batch <= 16) else 8
        cmd_yolo11m = f"python scripts/training/train_yolo_seg.py --model yolo11m-seg --dataset ./dataset --epochs {args.epochs} --batch {yolo_b} --workers {args.workers} --output_dir outputs/yolo11m-seg"
        cmd_mrcnn = f"python scripts/training/train_maskrcnn.py --dataset ./dataset --epochs {args.epochs} --batch 2 --num_workers {args.workers} --output_dir outputs/maskrcnn"
        cmd_m2f = f"python scripts/training/train_mask2former.py --dataset ./dataset --epochs {args.epochs} --batch 2 --num_workers {args.workers} --output_dir outputs/mask2former"

        command_string = f"{copy_cmd} && {cmd_yolo11m} && {cmd_mrcnn} && {cmd_m2f} && {cmd_export}"

    elif args.model == "yolo11m-seg":
        command_string = (
            f"{copy_cmd} && "
            f"python scripts/training/train_yolo_seg.py "
            f"--model yolo11m-seg "
            f"--dataset ./dataset "
            f"--epochs {args.epochs} "
            f"--batch {args.batch} "
            f"--workers {args.workers} "
            f"--output_dir outputs/yolo11m-seg && {cmd_export}"
        )
    elif args.model == "maskrcnn":
        command_string = f"{copy_cmd} && python scripts/training/train_maskrcnn.py --dataset ./dataset --epochs {args.epochs} --batch {args.batch} --num_workers {args.workers} --output_dir outputs/maskrcnn && {cmd_export}"
    elif args.model == "mask2former":
        command_string = f"{copy_cmd} && python scripts/training/train_mask2former.py --dataset ./dataset --epochs {args.epochs} --batch {args.batch} --num_workers {args.workers} --output_dir outputs/mask2former && {cmd_export}"

    # 5. Create the job display name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    display_name = f"{args.model}_v{dataset_version}_{timestamp}"

    print(f"\n  [STEP 4/4] Preparing Azure ML Job: {display_name}")
    print(f"             Experiment: {EXPERIMENT_NAME}")
    print(f"             Compute: {AZURE_COMPUTE_NAME}")
    print(f"             Command: {command_string}")

    data_asset = ml_client.data.get(name=AZURE_DATASET_NAME, version=dataset_version)

    job = command(
        code=code_dir,  # Upload clean isolated code snapshot
        command=command_string,
        inputs={
            "dataset": Input(
                type=AssetTypes.URI_FOLDER,
                path=data_asset.path,
                mode=InputOutputModes.DOWNLOAD,  # bulk copy to VM SSD (~500 MB/s) vs FUSE mount (0.2 MB/s)
            )
        },
        environment=env,
        compute=AZURE_COMPUTE_NAME,
        experiment_name=EXPERIMENT_NAME,
        display_name=display_name,
    )

    print("\n  [STEP 4/4] Submitting Job to Azure ML Compute...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"\n  [OK] Job submitted successfully to Azure ML!")
    print(f"  [LINK] Studio Web URL: {returned_job.studio_url}\n")
    print("=" * 70)
    print("  [STREAMING] Live training logs (Ctrl+C to stop watching,")
    print("              job will keep running on Azure ML in the background)")
    print("=" * 70)
    try:
        import io
        class FilteredStream(io.TextIOWrapper):
            def __init__(self, buffer):
                super().__init__(buffer, encoding='utf-8', errors='replace')
            def write(self, s):
                # Filter out massive 50-line Docker env var dumps (-e AZUREML_...)
                if "AZUREML_CURRENT_CLOUD_METADATA" in s or "MLFLOW_TRACKING_TOKEN" in s or "AZUREML_CONTEXT_MANAGER" in s:
                    return len(s)
                return super().write(s)

        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = FilteredStream(sys.stdout.buffer)
        ml_client.jobs.stream(returned_job.name)
    except KeyboardInterrupt:
        print("\n\n  [INFO] Log streaming stopped. Your job is STILL running on Azure ML!")
        print(f"  [LINK] Track it here: {returned_job.studio_url}\n")
        return
    except Exception as e:
        print(f"\n  [WARN] Live streaming stream disconnect: {e}")
        print(f"  [INFO] Falling back to polling job status...")

    # Wait for job completion if it was not interrupted
    print("  [*] Waiting for job to reach final status on Azure ML...")
    import time
    final_job = ml_client.jobs.get(returned_job.name)
    while final_job.status in ["Starting", "Preparing", "Running", "Queued"]:
        time.sleep(10)
        try:
            final_job = ml_client.jobs.get(returned_job.name)
        except Exception:
            pass

    print(f"  [STATUS] Final Azure ML Job Status: {final_job.status}")
    if final_job.status == "Failed":
        print(f"  [ERROR] Job '{returned_job.name}' failed on Azure ML Compute.")
        print(f"  [*] Attempting to fetch execution log from Azure ML...")
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_log_dir:
                ml_client.jobs.download(returned_job.name, download_path=tmp_log_dir)
                found_log = False
                for root, _, files in os.walk(tmp_log_dir):
                    for f in files:
                        if "driver" in f.lower() or f.endswith(".txt") or f.endswith(".log"):
                            p = os.path.join(root, f)
                            lines = open(p, encoding="utf-8", errors="replace").readlines()
                            if lines:
                                found_log = True
                                print(f"\n--- EXECUTION LOG ({f}, last 30 lines) ---")
                                for line in lines[-30:]:
                                    print("  " + line.rstrip())
                                print("-------------------------------------------\n")
                if not found_log:
                    print("  [INFO] No log files retrieved from default download artifact.")
        except Exception as e:
            print(f"  [WARN] Could not retrieve error log automatically: {e}")
        print(f"  [LINK] Check error details here: {returned_job.studio_url}")
        sys.exit(1)
    elif final_job.status == "Completed":
        print(f"  [OK] Job '{returned_job.name}' finished cleanly on Azure ML!")


if __name__ == "__main__":
    main()
