from huggingface_hub import hf_hub_download
import os
import torch

# Sources for TASK_SET_MT30:
# 1) Obs and act dims: https://www.tdmpc2.com/dataset
# 2) Order of tasks: https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/__init__.py
TASK_SET_MT30 = [
    # 19 original dmcontrol tasks
    ("walker-stand", 24, 6),
    ("walker-walk", 24, 6),
    ("walker-run", 24, 6),
    ("cheetah-run", 17, 6),
    ("reacher-easy", 6, 2),
    ("reacher-hard", 6, 2),
    ("acrobot-swingup", 6, 1),
    ("pendulum-swingup", 3, 1),
    ("cartpole-balance", 5, 1),
    ("cartpole-balance-sparse", 5, 1),
    ("cartpole-swingup", 5, 1),
    ("cartpole-swingup-sparse", 5, 1),
    ("cup-catch", 8, 2),
    ("finger-spin", 9, 2),
    ("finger-turn-easy", 12, 2),
    ("finger-turn-hard", 12, 2),
    ("fish-swim", 24, 5),
    ("hopper-stand", 15, 4),
    ("hopper-hop", 15, 4),
    # NOTE: There are 11 more (custom) dmcontrol tasks, we'll ignore them for now
]
MT30_ORIGINAL_DIR = os.path.join(os.environ.get("WRKDIR"), "data", "mt30", "original")
MT30_PER_TASK_DIR = os.path.join(os.environ.get("WRKDIR"), "data", "mt30", "per-task")
FILE_NAMES = ["chunk_0.pt", "chunk_1.pt", "chunk_2.pt", "chunk_3.pt"]

os.makedirs(MT30_ORIGINAL_DIR, exist_ok=True)
os.makedirs(MT30_PER_TASK_DIR, exist_ok=True)

print("Download TD-MPC2 dataset if necessary...")
for file_name in FILE_NAMES:
    if file_name not in os.listdir(MT30_ORIGINAL_DIR):
        file_path = hf_hub_download(
            repo_id="nicklashansen/tdmpc2",
            filename=f"mt30/{file_name}",
            local_dir=MT30_ORIGINAL_DIR,
            repo_type="dataset",
        )
        print(f"Downloaded {file_name} to {file_path}")

print("Split TD-MPC2 dataset by task name...")
data_per_task = {}
for file_name in FILE_NAMES:
    file_path = os.path.join(MT30_ORIGINAL_DIR, file_name)
    print(f"Reading '{file_path}'...")
    chunk = torch.load(file_path, weights_only=False)
    for td_episode in chunk:
        task_id = td_episode["task"][0]
        if task_id >= len(TASK_SET_MT30):
            continue  # Custom dmcontrol tasks are ignored for now
        task_info = TASK_SET_MT30[task_id]
        del td_episode["task"]  # No longer needed, we have task_name now

        if task_info not in data_per_task:
            data_per_task[task_info] = []
        data_per_task[task_info].append(td_episode)

for task_info, tds in data_per_task.items():
    task_name, obs_dim, act_dim = task_info
    file_path = os.path.join(MT30_PER_TASK_DIR, f"{task_name}.pt")
    td = torch.stack(tds)
    td["obs"] = td["obs"][..., :obs_dim]
    td["action"] = td["action"][..., :act_dim]
    torch.save(td, file_path)
    print(f"Saved '{task_name}' data to '{file_path}'")
