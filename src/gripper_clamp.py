"""Gripper clamp (stall) detection for the SO-101.

Gripper convention: 0 = closed, 100 = open. A clamp is when the gripper is
commanded closed but the follower has stopped moving because it is squeezing
an object. Used as the grasp / release identifier for RoboTAP stages.
"""

from __future__ import annotations


class GripperClampDetector:
    def __init__(
        self,
        close_margin: float = 1.0,
        motion_eps: float = 0.5,
        stall_frames: int = 3,
        release_margin: float = 3.0,
        verbose: bool = True,
        label: str = "",
    ):
        self.close_margin = float(close_margin)
        self.motion_eps = float(motion_eps)
        self.stall_frames = int(stall_frames)
        self.release_margin = float(release_margin)
        self.verbose = verbose
        self.label = label
        self.reset()

    def reset(self, measured: float | None = None) -> None:
        self.clamped = False
        self.clamp_value: float | None = None
        self._prev: float | None = measured
        self._stall_count = 0

    @property
    def stalling(self) -> bool:
        """Commanded closed and follower stopped for enough frames."""
        return self._stall_count >= self.stall_frames

    def update(
        self, commanded: float, measured: float, frame: int | None = None
    ) -> str | None:
        """Feed one frame. Returns "clamp", "release", or None."""
        commanded = float(commanded)
        measured = float(measured)
        prev = measured if self._prev is None else self._prev
        closing = commanded < measured - self.close_margin
        moved_closed = (prev - measured) > self.motion_eps
        if closing and not moved_closed:
            self._stall_count += 1
        else:
            self._stall_count = 0
        self._prev = measured

        event = None
        if not self.clamped and self.stalling:
            self.clamped = True
            self.clamp_value = measured
            event = "clamp"
        elif self.clamped and commanded > measured + self.release_margin:
            self.clamped = False
            self.clamp_value = None
            event = "release"

        if event and self.verbose:
            tag = "[CLAMP]" if event == "clamp" else "[RELEASE]"
            where = f" frame={frame}" if frame is not None else ""
            prefix = f"{self.label} " if self.label else ""
            print(
                f"{prefix}{tag}{where} grip={measured:.1f} cmd={commanded:.1f}",
                flush=True,
            )
        return event
