#!/usr/bin/env python3
from typing import Optional

from torchrl.envs import DMControlEnv, DMControlWrapper
from torchrl.envs.transforms import CatTensors, TransformedEnv


def make_env(
    env_name: str,
    task_name: Optional[str] = None,
    from_pixels: bool = True,
    frame_skip: int = 2,
    pixels_only: bool = False,
    record_video: bool = False,
    device: str = "cpu",
):
    if env_name == "swimmer" and task_name not in ["swimmer6", "swimmer15"]:
        print(f"Create custom swimmer task: {task_name}")
        # Create swimmer env with custom link count
        from dm_control.suite.swimmer import swimmer

        n_links = int(task_name.split("-")[-1])  # ("swimmer", "swim-{n_links}")
        custom_swimmer_env = swimmer(n_links)
        env = DMControlWrapper(
            custom_swimmer_env,
            from_pixels=from_pixels or record_video,
            frame_skip=frame_skip,
            pixels_only=pixels_only,
            device=device,
        )
    else:
        if env_name == "cup":
            env_name = "ball_in_cup"

        print(f"Create standard {env_name} task: {task_name}")
        env = DMControlEnv(
            env_name=env_name,
            task_name=task_name,
            from_pixels=from_pixels or record_video,
            frame_skip=frame_skip,
            pixels_only=pixels_only,
            device=device,
        )
    if not pixels_only:
        # Put "position"/"velocity"/"orientation" into "observation"
        obs_keys = [key for key in env.observation_spec.keys()]
        if from_pixels or record_video:
            obs_keys.remove("pixels")
        env = TransformedEnv(env, CatTensors(in_keys=obs_keys, out_key="observation"))

        # env = TransformedEnv(
        #     env,
        #     CatTensors(in_keys=obs_keys, out_key=("observation", "state")),
        # )
    return env
