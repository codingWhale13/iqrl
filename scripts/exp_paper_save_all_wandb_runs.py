import os
import json
import wandb
import numpy as np
import pandas as pd


# === Constants ===
ENTITY_AND_PROJECT = "kielen1-aalto-university/iqrl"


os.makedirs("wandb_backup", exist_ok=True)
wandb.login()
api = wandb.Api()
all_runs = api.runs("kielen1-aalto-university/iqrl")
for run in all_runs:
    run_dir = os.path.join("wandb_backup", run.id)
    os.makedirs(run_dir, exist_ok=True)

    # Save summary
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(run.summary._json_dict, fp=f, indent=2)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(run.config, fp=f, indent=2)

    try:
        # Save run metadata
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(
                {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": str(run.created_at),
                    "tags": run.tags,
                    "notes": run.notes,
                },
                f,
                indent=2,
            )
        # Download all associated files (e.g. models, images, logs)
        for file in run.files():
            if file == "checkpoint":
                breakpoint()
            if "video" in file.name:
                continue
            file.download(run_dir, replace=True)
    except:
        print("smth went wrong; continue")

    # Save full metric history
    try:
        history = run.history(samples=1000000)  # increase if needed
        history.to_csv(os.path.join(run_dir, "history.csv"), index=False)
    except Exception as e:
        print(f"⚠️ Failed to get history for run {run.id}: {e}")

    print(f"✅ Downloaded run {run.id} to {run_dir}")
