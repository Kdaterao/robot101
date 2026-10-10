
import shutil
from lerobot.utils.constants import HF_LEROBOT_HOME


def main():
    repo_id = "kdaterao/so101_data2"
    dataset_root = HF_LEROBOT_HOME / repo_id

    shutil.rmtree(dataset_root)


if __name__ == "__main__":
        main()
