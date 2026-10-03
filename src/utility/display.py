import cv2
import numpy as np
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

from .robot import FPS

_cv_windows = set()
_refocused_pygame = False


def preview_camera(index: int) -> None:
    """Preview a camera with the same capture pipeline the SO101 uses."""
    camera = OpenCVCamera(
        OpenCVCameraConfig(index_or_path=index, width=640, height=480, fps=FPS)
    )
    camera.connect()
    window = f"camera {index} | press q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = np.asarray(camera.read())
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()


def focus_pygame_window() -> None:
    """Bring the pygame/SDL window to the front so Xbox input is received on macOS."""
    import pygame

    pygame.event.pump()
    try:
        from pygame._sdl2.video import Window

        Window.from_display_module().focus()
    except Exception:
        pass
    pygame.display.flip()


def show_cameras(obs: dict, camera_key=None) -> None:
    """Display robot camera frames in OpenCV windows."""
    global _refocused_pygame

    for key, value in obs.items():
        if key != camera_key and camera_key:
            continue
        if not hasattr(value, "ndim") or value.ndim != 3:
            continue
        frame = np.asarray(value)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        name = str(key)
        if name not in _cv_windows:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            _cv_windows.add(name)
        cv2.imshow(name, frame)
    cv2.waitKey(1)
    if _cv_windows and not _refocused_pygame:
        focus_pygame_window()
        _refocused_pygame = True


def ask_question(pygame, screen, question):
    font = pygame.font.Font(None, 32)
    clock = pygame.time.Clock()
    text = ""
    pygame.event.clear()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return text
                if event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                else:
                    text += event.unicode
        screen.fill((30, 30, 30))
        question_surface = font.render(question, True, (255, 255, 255))
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(question_surface, (20, 10))
        screen.blit(text_surface, (20, 50))
        pygame.display.flip()
        clock.tick(30)
