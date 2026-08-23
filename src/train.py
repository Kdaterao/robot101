from pathlib import Path

import torch
from huggingface_hub import repo_exists
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.utils.constants import HF_LEROBOT_HOME


def main():
    #--------------------
    #  SOURCE DEFINITION
    #--------------------

    repo = "kdaterao/smolVLA_desk"
    ds_repo = "kdaterao/so101_data"
    output_directory = Path(__file__).resolve().parent.parent / "outputs" / "train" / "smolvla_desk"
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    #-------------------
    #  LOAD IN MODEL
    #-------------------

    local_ckpt = output_directory / "model.safetensors"
    
    if local_ckpt.exists():
        pretrained_path = str(output_directory)
        print(f"Resuming from local checkpoint {output_directory}")
    elif repo_exists(repo):
        pretrained_path = repo
        print(f"Loading from Hub {repo}")
    else:
        pretrained_path = "lerobot/smolvla_base"
        print(f"Starting from {pretrained_path}")

    policy = SmolVLAPolicy.from_pretrained(pretrained_path)
    cfg = policy.config
    policy.train()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    #------------------------
    #   LOAD IN DATASET
    #------------------------

    # check if we have local copy of dataset
    dataset_root = HF_LEROBOT_HOME / ds_repo
    dataset_kwargs = {}
    if (dataset_root / "meta").exists():
        print(f"Loading local dataset at {dataset_root}")
        dataset_kwargs["root"] = dataset_root
    else:
        print(f"Loading dataset from Hub: {ds_repo}")

    # load in dataset metadata
    dataset_metadata = LeRobotDatasetMetadata(ds_repo, **dataset_kwargs)

    # create delta timestamps for the dataset
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.images.camera1": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.images.camera2": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # create dataset object
    dataset = LeRobotDataset(ds_repo, delta_timestamps=delta_timestamps, **dataset_kwargs)

    #-----------------------
    #  TRAINING DEFINITIONS
    #-----------------------
    #change here!
    training_steps = 20000  # SmolVLA docs often use ~20k; 500 just a test
    log_freq = 50
    batch_size = 16


    # create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    #-----------------------
    #  SAVE
    #-----------------------

    # save our checkpoint and pre/postprocessors to the output directory (in case huggingface fails to push)
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"Saved checkpoint to {output_directory}")

    # push our checkpoint and pre/postprocessors to the hub.
    try:
        policy.push_to_hub(repo_id=repo, private=False)
        preprocessor.push_to_hub(repo)
        postprocessor.push_to_hub(repo)
        print(f"Pushed to Hub {repo}")
    except Exception as exc:
        print(f"Hub push failed (local checkpoint is at {output_directory}): {exc}")


if __name__ == "__main__":
    main()
