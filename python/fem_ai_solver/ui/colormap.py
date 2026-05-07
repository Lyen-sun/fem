from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor


@dataclass(frozen=True, slots=True)
class Colormap:
    """Simple lightweight engineering colormap (blue -> cyan -> green -> yellow -> red)."""

    def _interp(self, c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> QColor:
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        return QColor(r, g, b)

    def color(self, value: float, vmin: float, vmax: float) -> QColor:
        if vmax <= vmin:
            return QColor(52, 152, 219)

        ratio = (value - vmin) / (vmax - vmin)
        ratio = max(0.0, min(1.0, ratio))

        anchors = [
            (0.00, (33, 102, 172)),
            (0.25, (67, 162, 202)),
            (0.50, (65, 171, 93)),
            (0.75, (241, 196, 15)),
            (1.00, (192, 57, 43)),
        ]

        for index in range(len(anchors) - 1):
            left_pos, left_color = anchors[index]
            right_pos, right_color = anchors[index + 1]
            if left_pos <= ratio <= right_pos:
                local_t = (ratio - left_pos) / (right_pos - left_pos)
                return self._interp(left_color, right_color, local_t)

        return QColor(*anchors[-1][1])
