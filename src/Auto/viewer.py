import glfw
import mujoco

# NOTE: glfw is the goto option for mujcoo windows (muchj like how it is for opengl)


class MainThreadViewer:
    """Minimal MuJoCo window that runs on the Python main thread.

    launch_passive needs mjpython on macOS, and mjpython cannot find uv's
    libpython. This viewer avoids that by rendering in the control loop.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, key_callback=None):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        self.model = model
        self.data = data
        self._key_callback = key_callback
        self._button_left = False
        self._button_right = False
        self._last_x = 0.0
        self._last_y = 0.0

        glfw.window_hint(glfw.SAMPLES, 4)
        self.window = glfw.create_window(1280, 800, "SO101 real2sim", None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Could not create a GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.cam = mujoco.MjvCamera()
        if hasattr(mujoco, "mjv_defaultFreeCamera"):
            mujoco.mjv_defaultFreeCamera(model, self.cam)
        else:
            mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[:] = (0.1, 0.0, 0.12)
        self.cam.distance = 0.7
        self.cam.azimuth = 160
        self.cam.elevation = -20

        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt)
        self.scn = mujoco.MjvScene(model, maxgeom=20000)
        self.con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.overlay = ""

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS or self._key_callback is None:
            return
        self._key_callback(key)

    def _on_mouse_button(self, window, button, act, mods):
        self._button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        x, y = glfw.get_cursor_pos(window)
        self._last_x, self._last_y = x, y

    def _on_mouse_move(self, window, xpos, ypos):
        dx = xpos - self._last_x
        dy = ypos - self._last_y
        self._last_x, self._last_y = xpos, ypos
        if not (self._button_left or self._button_right):
            return

        width, height = glfw.get_window_size(window)
        if height == 0:
            return
        shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        if self._button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        self._move_camera(action, dx / height, dy / height)

    def _on_scroll(self, window, xoffset, yoffset):
        self._move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset)

    def _move_camera(self, action, reldx, reldy):
        # MuJoCo 3.12 dropped the scene argument from mjv_moveCamera.
        mujoco.mjv_moveCamera(self.model, int(action), float(reldx), float(reldy), self.cam)

    def is_running(self) -> bool:
        return not glfw.window_should_close(self.window)

    def sync(self) -> None:
        if not self.is_running():
            return
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )
        mujoco.mjr_render(viewport, self.scn, self.con)
        if self.overlay:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                self.overlay,
                "",
                self.con,
            )
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self) -> None:
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()
