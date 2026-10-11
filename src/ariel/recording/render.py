"""Render recorded body points and obstacle boxes in a lunar palette.

No simulator GUI is needed. Output is HxWx3 uint8; both dimensions must be
multiples of 16.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SKY = (12, 14, 24)
GROUND = (96, 96, 104)
OBSTACLE = (140, 136, 128)
BODY = (236, 176, 64)
TEXT_BAND = (24, 28, 44)


@dataclass(frozen=True)
class View:
    """World rectangle shown in the frame; the camera can follow the body in x."""

    width_px: int = 512
    height_px: int = 512
    world_width: float = 1.0
    world_height: float = 1.0
    follow: bool = True


def _to_pixels(points: np.ndarray, x0: float, view: View) -> tuple[np.ndarray, np.ndarray]:
    sx = view.width_px / view.world_width
    sy = view.height_px / view.world_height
    px = ((points[:, 0] - x0) * sx).astype(np.int64)
    py = (view.height_px - 1 - points[:, 1] * sy).astype(np.int64)
    return px, py


def render_frame(points: np.ndarray, course: dict | None = None, view: View = View(),
                 point_radius: int = 1) -> np.ndarray:
    """points: [P, 2] world coordinates (x right, y up). course: {"ground_y", "obstacles": [[x0,y0,x1,y1], ...]}."""
    if view.width_px % 16 or view.height_px % 16:
        raise ValueError("frame size must be a multiple of 16")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    course = course or {}
    frame = np.empty((view.height_px, view.width_px, 3), dtype=np.uint8)
    frame[:] = SKY
    x0 = 0.0
    if view.follow and len(points):
        x0 = float(points[:, 0].mean()) - view.world_width / 2
    sy = view.height_px / view.world_height
    sx = view.width_px / view.world_width
    ground_y = course.get("ground_y")
    if ground_y is not None:
        row = int(view.height_px - ground_y * sy)
        frame[max(row, 0):] = GROUND
    for box in course.get("obstacles", []):
        bx0, by0, bx1, by1 = box
        c0, c1 = int((bx0 - x0) * sx), int((bx1 - x0) * sx)
        r0, r1 = int(view.height_px - by1 * sy), int(view.height_px - by0 * sy)
        c0, c1 = max(c0, 0), min(c1, view.width_px)
        r0, r1 = max(r0, 0), min(r1, view.height_px)
        if c0 < c1 and r0 < r1:
            frame[r0:r1, c0:c1] = OBSTACLE
    if len(points):
        px, py = _to_pixels(points, x0, view)
        for dy in range(-point_radius, point_radius + 1):
            for dx in range(-point_radius, point_radius + 1):
                xs, ys = px + dx, py + dy
                ok = (xs >= 0) & (xs < view.width_px) & (ys >= 0) & (ys < view.height_px)
                frame[ys[ok], xs[ok]] = BODY
    return frame
