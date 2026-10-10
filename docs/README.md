# Run guides

See [source layout](../src/README.md) for the organized Python package.

Run commands from the repository root unless a guide explicitly changes directory.
Package installers live in `scripts/setup/`; Python entrypoints live in `src/`
and `scripts/`. Model downloads, processing, training and uploads are explicit
steps in the guides.

| Workflow | Installer | Run guide |
| --- | --- | --- |
| Ubuntu / A6000 episode preprocessing | `scripts/setup/linux_setup.sh` | [Preprocessing](preprocessing.md) |
| Molmo2-4B connector training | `scripts/setup/linux_setup.sh --profile molmo2` | [Training](molmo2_training.md) |
| Local viewers | `requirements.txt` viewer profile | [Viewing](viewing.md) |
| Robot control / SmolVLA | `scripts/setup/linux_setup.sh --profile robot` | [Project README](../README.md) |
| Older TAPIR workflows | `scripts/setup/linux_setup.sh` / `windows_setup.ps1` | [TAPIR commands](legacy/TAPNET_COMMANDS.md) |
| Point label collection | `requirements.txt` labeling profile | [Molmo gripper point dataset](molmo_gripper_point_data.md) |

Short workflow guides:

- [Molmo gripper point dataset](molmo_gripper_point_data.md)
- [Grounded robot training data](grounded_training_data.md)

There are two setup entrypoints: `linux_setup.sh` for Ubuntu and
`windows_setup.ps1` for native Windows PowerShell. Linux defaults to preprocessing;
use `--profile molmo2`, `--profile robot`, or `--profile all` to prepare the other
environments. They stay isolated because their dependency versions differ.
Windows keeps the original lean TAPIR/CUDA setup in `.venv`. All repository-owned dependency sets live in one `requirements.txt`, including
`einops` and `einshape`. Select a profile with:

```bash
python scripts/install_requirements.py --profile viewer
```

Profiles: `grounded`, `viewer`, `tapnet`, `robot`, `labeling`,
`molmo-grounding`. The installers select their profiles
for you. A plain `pip install -r requirements.txt` installs `grounded`.
Legacy Transformers pins remain in separate environments.
The old generated robot requirements are replaced by LeRobot's declared extras;
`uv.lock` remains the project lockfile. Upstream Molmo training dependencies
come from the editable upstream training package.

## Linux setup

```bash
bash scripts/setup/linux_setup.sh
source .venv-grounded/bin/activate
hf auth login
```

To prepare preprocessing, Molmo2 training and robot environments together:

```bash
bash scripts/setup/linux_setup.sh --profile all
```

## Windows setup

Run in PowerShell from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup/windows_setup.ps1
.\.venv\Scripts\Activate.ps1
```

This preserves the TAPIR setup with CUDA 12.4 PyTorch. Download model checkpoints
separately using the run guides.

The root `packages*.sh` installers moved into `scripts/setup/`. The former
preprocessing, data preparation and training Bash launchers were replaced by
these documented Python commands. Run settings are command arguments.
