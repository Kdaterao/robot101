"""Camera preview helpers.

Uses pygame for windows — opencv-python-headless has no highgui
(cv2.imshow / namedWindow raise on this install).
"""

from __future__ import annotations

import cv2
import numpy as np
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

from .helpers import FPS

_pygame_surfaces: dict[str, object] = {}


def _ensure_pygame():
    import pygame

    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()
    return pygame


def _frame_to_rgb(value) -> np.ndarray:
    frame = np.asarray(value)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 frame, got {frame.shape}")
    # LeRobot OpenCV camera observations are RGB.
    return np.ascontiguousarray(frame)


def _blit_rgb(pygame, name: str, rgb: np.ndarray, max_w: int = 960) -> None:
    h, w = rgb.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        rgb = cv2.resize(
            rgb, (max_w, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA
        )
        h, w = rgb.shape[:2]
    rgb = np.ascontiguousarray(rgb)
    surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
    screen = _pygame_surfaces.get(name)
    if screen is None or screen.get_size() != (w, h):
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(name)
        _pygame_surfaces[name] = screen
    screen.blit(surf, (0, 0))
    pygame.display.flip()


def preview_camera(index: int) -> None:
    """Preview a camera with the same capture pipeline the SO101 uses."""
    pygame = _ensure_pygame()
    camera = OpenCVCamera(
        OpenCVCameraConfig(index_or_path=index, width=640, height=480, fps=FPS)
    )
    camera.connect()
    window = f"camera {index} | press q to quit"
    try:
        running = True
        while running:
            frame = _frame_to_rgb(camera.read())
            _blit_rgb(pygame, window, frame)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False
    finally:
        camera.disconnect()
        _pygame_surfaces.pop(window, None)
        pygame.display.quit()


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
    """Display robot camera frames in a pygame window."""
    pygame = _ensure_pygame()
    shown = False
    for key, value in obs.items():
        if camera_key is not None and key != camera_key:
            continue
        if not hasattr(value, "ndim") or value.ndim != 3:
            continue
        try:
            rgb = _frame_to_rgb(value)
        except ValueError:
            continue
        _blit_rgb(pygame, str(key), rgb)
        shown = True
        # Single shared display surface: show first matching camera only.
        break
    if shown:
        pygame.event.pump()


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
