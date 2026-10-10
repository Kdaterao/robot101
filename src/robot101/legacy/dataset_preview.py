from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

root = HF_LEROBOT_HOME / "kdaterao/so101_data"

ds = LeRobotDataset(
    repo_id="kdaterao/so101_data",
    root=root,
)

print("episodes:", ds.num_episodes)
print("frames:", ds.num_frames)

for ep in range(ds.num_episodes):
    print("\nEPISODE", ep)
    print(ds.meta.episodes[ep])




import cv2
from pathlib import Path

video = Path.home() / ".cache/huggingface/lerobot/kdaterao/so101_data/videos/observation.images.camera1/chunk-000/file-011.mp4"

cap = cv2.VideoCapture(str(video))

print("opened:", cap.isOpened())
print("reported frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("reported FPS:", cap.get(cv2.CAP_PROP_FPS))
print("reported duration:", cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))

actual = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    actual += 1

cap.release()

print("ACTUAL decoded frames:", actual)