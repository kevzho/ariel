"""Crash-tolerant episode video: fragmented H.264 MP4 segments via imageio-ffmpeg.

Fragmented MP4 (`+frag_keyframe+empty_moov+default_base_moof`) with
`-flush_packets 1` stays decodable if the writer is killed: at most about one
GOP (`-g fps`, ~1 s) is lost. A plain MP4 is unreadable without its trailer.
Long episodes roll over to a new segment every `segment_frames` frames, which
also bounds the loss.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from .ledger import RecordingError


def output_params(fps: int) -> list[str]:
    # -bf 0: no B-frames, so frame k is presented at exactly k / fps (browser sync).
    return ["-preset", "veryfast", "-crf", "23", "-g", str(fps), "-bf", "0", "-pix_fmt", "yuv420p",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-flush_packets", "1"]

class SegmentedVideoWriter:
    def __init__(self, directory: Path, stem: str, *, fps: int = 30, segment_frames: int = 900):
        self.directory = Path(directory)
        self.stem = stem
        self.fps = fps
        self.segment_frames = segment_frames
        self.segments: list[dict] = []        # {"path", "first_frame", "frames"}
        self._writer = None
        self._size = None
        self.frames = 0

    def _open_segment(self) -> None:
        import imageio_ffmpeg

        path = self.directory / f"{self.stem}.seg{len(self.segments):03d}.mp4"
        try:
            self._writer = imageio_ffmpeg.write_frames(
                str(path), self._size, fps=self.fps, codec="libx264", quality=None,
                macro_block_size=16, output_params=output_params(self.fps), ffmpeg_log_level="error")
            self._writer.send(None)
        except Exception as exc:            # noqa: BLE001 - any encoder start failure stops training
            raise RecordingError(f"could not start video segment {path}: {exc}") from exc
        self.segments.append({"path": path.name, "first_frame": self.frames, "frames": 0})

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RecordingError(f"frames must be HxWx3 uint8, got {frame.shape}")
        height, width = frame.shape[:2]
        if width % 16 or height % 16:
            raise RecordingError(f"frame size {width}x{height} must be a multiple of 16")
        if self._size is None:
            self._size = (width, height)
        elif self._size != (width, height):
            raise RecordingError("frame size changed within an episode")
        if self._writer is None or self.segments[-1]["frames"] >= self.segment_frames:
            self.close_segment()
            self._open_segment()
        try:
            self._writer.send(frame)
        except Exception as exc:            # noqa: BLE001
            raise RecordingError(f"video write failed: {exc}") from exc
        self.segments[-1]["frames"] += 1
        self.frames += 1

    def close_segment(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:        # noqa: BLE001
                raise RecordingError(f"video finalize failed: {exc}") from exc
            self._writer = None

    def close(self) -> list[dict]:
        self.close_segment()
        return self.segments
