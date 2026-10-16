"""Air Drawing Module.

Provides smooth continuous line drawing on a virtual canvas, multiple vibrant colors,
brush and eraser sizing, canvas clearing, snapshot saving, and gesture-based control.
"""

from typing import List, Tuple, Optional, Dict, Any
import os
import time
from datetime import datetime
import cv2
import numpy as np


class AirDrawer:
    """Manages virtual air drawing canvas, drawing gestures, and toolbar controls.

    Attributes:
        width (int): Canvas width in pixels.
        height (int): Canvas height in pixels.
        output_dir (str): Directory where snapshots are saved.
    """

    # Color definitions (BGR format for OpenCV)
    COLORS: Dict[str, Tuple[int, int, int]] = {
        "Cyan": (255, 255, 0),
        "Green": (0, 255, 0),
        "Red": (0, 0, 255),
        "Orange": (0, 140, 255),
        "Magenta": (255, 0, 255),
        "Yellow": (0, 230, 255),
        "White": (255, 255, 255),
    }

    BRUSH_SIZES = [4, 8, 14, 22]
    ERASER_SIZE = 50

    def __init__(
        self,
        width: int = 1536,
        height: int = 864,
        output_dir: str = "output",
    ):
        self.width = width
        self.height = height
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Virtual drawing canvas (3 channels BGR)
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Drawing state
        self.color_names = list(self.COLORS.keys())
        self.current_color_idx = 0
        self.brush_size_idx = 1  # default 8px
        self.is_eraser = False

        self.previous_point: Optional[Tuple[int, int]] = None
        self.smoothed_point: Optional[Tuple[int, int]] = None

        # Notification banner state
        self.notification_msg: str = ""
        self.notification_time: float = 0.0

        # Toolbar button layout
        self.toolbar_height = 70
        self.buttons: List[Dict[str, Any]] = []
        self._init_toolbar()

        # Click / selection cooldown to prevent repeated rapid clicks
        self.last_action_time: float = 0.0
        self.action_cooldown: float = 0.40

    def resize_canvas(self, new_width: int, new_height: int) -> None:
        """Adapts canvas dimensions if display resolution changes."""
        if (new_width, new_height) != (self.width, self.height):
            resized = cv2.resize(self.canvas, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            self.width = new_width
            self.height = new_height
            self.canvas = resized
            self._init_toolbar()

    def _init_toolbar(self) -> None:
        """Sets up non-overlapping interactive toolbar buttons."""
        self.buttons = []
        btn_y1 = 12
        btn_y2 = 60

        # Palette starts after the mode switcher buttons (at x = 320)
        palette_x = 320
        color_btn_w = 44
        gap = 10

        # Color circles
        for i, name in enumerate(self.color_names):
            x1 = palette_x + i * (color_btn_w + gap)
            x2 = x1 + color_btn_w
            self.buttons.append({
                "type": "color",
                "label": name,
                "color": self.COLORS[name],
                "rect": (x1, btn_y1, x2, btn_y2),
                "index": i,
            })

        # Eraser button
        eraser_x = palette_x + len(self.color_names) * (color_btn_w + gap) + 15
        eraser_w = 88
        self.buttons.append({
            "type": "eraser",
            "label": "ERASER",
            "rect": (eraser_x, btn_y1, eraser_x + eraser_w, btn_y2),
        })

        # Brush size toggle button
        size_x = eraser_x + eraser_w + gap
        size_w = 84
        self.buttons.append({
            "type": "brush_size",
            "label": "SIZE",
            "rect": (size_x, btn_y1, size_x + size_w, btn_y2),
        })

        # Clear canvas button
        clear_x = size_x + size_w + gap
        clear_w = 84
        self.buttons.append({
            "type": "clear",
            "label": "CLEAR",
            "rect": (clear_x, btn_y1, clear_x + clear_w, btn_y2),
        })

        # Save snapshot button
        save_x = clear_x + clear_w + gap
        save_w = 84
        self.buttons.append({
            "type": "save",
            "label": "SAVE",
            "rect": (save_x, btn_y1, save_x + save_w, btn_y2),
        })

    def get_current_color(self) -> Tuple[int, int, int]:
        """Returns the active drawing color (or black for eraser)."""
        if self.is_eraser:
            return (0, 0, 0)
        color_name = self.color_names[self.current_color_idx]
        return self.COLORS[color_name]

    def get_current_thickness(self) -> int:
        """Returns the active stroke thickness."""
        if self.is_eraser:
            return self.ERASER_SIZE
        return self.BRUSH_SIZES[self.brush_size_idx]

    def clear_canvas(self) -> None:
        """Clears all strokes on the virtual canvas."""
        self.canvas[:] = 0
        self.previous_point = None
        self.smoothed_point = None
        self._notify("Canvas Cleared")

    def save_canvas(self, background_frame: Optional[np.ndarray] = None) -> str:
        """Saves current drawing as high-resolution PNG file into output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"air_drawing_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)

        if background_frame is not None:
            save_img = self.blend_with_canvas(background_frame)
        else:
            dark_bg = np.zeros_like(self.canvas)
            dark_bg[:] = (20, 20, 20)
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            inv_mask = cv2.bitwise_not(mask)
            bg_masked = cv2.bitwise_and(dark_bg, dark_bg, mask=inv_mask)
            save_img = cv2.add(bg_masked, self.canvas)

        cv2.imwrite(filepath, save_img)
        self._notify(f"Saved: {filepath}")
        return filepath

    def cycle_color(self) -> None:
        """Switches to the next color in the palette."""
        self.is_eraser = False
        self.current_color_idx = (self.current_color_idx + 1) % len(self.color_names)
        self._notify(f"Color: {self.color_names[self.current_color_idx]}")

    def cycle_brush_size(self) -> None:
        """Toggles through brush thickness options."""
        self.brush_size_idx = (self.brush_size_idx + 1) % len(self.BRUSH_SIZES)
        self._notify(f"Brush Size: {self.BRUSH_SIZES[self.brush_size_idx]}px")

    def toggle_eraser(self) -> None:
        """Toggles between eraser mode and drawing color mode."""
        self.is_eraser = not self.is_eraser
        status = "Eraser Enabled" if self.is_eraser else f"Color: {self.color_names[self.current_color_idx]}"
        self._notify(status)

    def _notify(self, message: str) -> None:
        """Sets temporary feedback notification banner."""
        self.notification_msg = message
        self.notification_time = time.time()

    def process_hand(
        self,
        frame: np.ndarray,
        index_tip: Optional[Tuple[int, int]],
        fingers: List[int],
        is_pinch: bool = False,
    ) -> str:
        """Processes hand landmarks to draw lines or interact with toolbar.

        Gestures:
            1. DRAW: Only Index finger up -> Continuous smooth drawing.
            2. PAUSE / HOVER: Index + Middle finger up -> Free cursor movement; no drawing.
            3. PINCH / HOVER: Clicks toolbar buttons when over the top bar.
        """
        if index_tip is None:
            self.previous_point = None
            self.smoothed_point = None
            return "NO_HAND"

        x, y = index_tip
        index_up = len(fingers) >= 2 and fingers[1] == 1
        middle_up = len(fingers) >= 3 and fingers[2] == 1
        now = time.time()

        in_toolbar = y <= (self.toolbar_height + 10)

        # 1. PAUSE / HOVER (Index and Middle fingers UP)
        if index_up and middle_up:
            self.previous_point = None
            # Draw cursor ring indicating hover/pause
            cv2.circle(frame, (x, y), 12, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)

            if in_toolbar:
                self._handle_toolbar_interaction(x, y, is_click=True)
                return "TOOLBAR"

            return "PAUSED"

        # 2. DRAWING MODE (Only Index finger UP, Middle finger DOWN)
        elif index_up and not middle_up and not in_toolbar:
            color = self.get_current_color()
            thickness = self.get_current_thickness()

            # Responsive Exponential Moving Average smoothing
            if self.smoothed_point is None:
                self.smoothed_point = (x, y)
            else:
                alpha = 0.70
                sx = int(alpha * x + (1 - alpha) * self.smoothed_point[0])
                sy = int(alpha * y + (1 - alpha) * self.smoothed_point[1])
                self.smoothed_point = (sx, sy)

            current_pt = self.smoothed_point

            if self.previous_point is not None:
                dist = np.hypot(current_pt[0] - self.previous_point[0], current_pt[1] - self.previous_point[1])
                if dist < 140:
                    cv2.line(self.canvas, self.previous_point, current_pt, color, thickness, cv2.LINE_AA)

            self.previous_point = current_pt

            if self.is_eraser:
                cv2.circle(frame, current_pt, self.ERASER_SIZE // 2, (200, 200, 200), 2, cv2.LINE_AA)
                cv2.circle(frame, current_pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                return "ERASING"
            else:
                cv2.circle(frame, current_pt, max(5, thickness // 2), color, -1, cv2.LINE_AA)
                cv2.circle(frame, current_pt, max(7, (thickness // 2) + 2), (255, 255, 255), 1, cv2.LINE_AA)
                return "DRAWING"

        # In toolbar with index up
        elif index_up and in_toolbar:
            self.previous_point = None
            cv2.circle(frame, (x, y), 9, (0, 255, 255), 2, cv2.LINE_AA)
            self._handle_toolbar_interaction(x, y, is_click=(is_pinch or (now - self.last_action_time > 0.5)))
            return "TOOLBAR"

        else:
            self.previous_point = None
            self.smoothed_point = None
            return "STANDBY"

    def _handle_toolbar_interaction(self, x: int, y: int, is_click: bool = False) -> None:
        """Handles hovering and clicking over toolbar buttons."""
        now = time.time()
        for btn in self.buttons:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                if is_click and (now - self.last_action_time > self.action_cooldown):
                    self.last_action_time = now
                    b_type = btn["type"]

                    if b_type == "color":
                        self.is_eraser = False
                        self.current_color_idx = btn["index"]
                        self._notify(f"Color: {btn['label']}")
                    elif b_type == "eraser":
                        self.toggle_eraser()
                    elif b_type == "brush_size":
                        self.cycle_brush_size()
                    elif b_type == "clear":
                        self.clear_canvas()
                    elif b_type == "save":
                        self.save_canvas()
                break

    def draw_ui(self, frame: np.ndarray, current_point: Optional[Tuple[int, int]] = None) -> None:
        """Renders toolbar buttons, active selection highlights, and temporary banners."""
        cx = current_point[0] if current_point else -1
        cy = current_point[1] if current_point else -1

        for btn in self.buttons:
            x1, y1, x2, y2 = btn["rect"]
            b_type = btn["type"]
            is_hovered = (x1 <= cx <= x2 and y1 <= cy <= y2)

            if b_type == "color":
                idx = btn["index"]
                is_active = (not self.is_eraser and idx == self.current_color_idx)

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                radius = 18 if not is_hovered else 20
                cv2.circle(frame, center, radius, btn["color"], -1, cv2.LINE_AA)

                if is_active:
                    cv2.circle(frame, center, radius + 4, (255, 255, 255), 2, cv2.LINE_AA)
                elif is_hovered:
                    cv2.circle(frame, center, radius + 3, (0, 240, 255), 2, cv2.LINE_AA)

            else:
                is_active = (b_type == "eraser" and self.is_eraser)
                bg_color = (0, 160, 230) if is_active else ((85, 85, 85) if is_hovered else (50, 50, 50))
                border_color = (255, 255, 255) if is_active else ((0, 230, 255) if is_hovered else (100, 100, 100))

                cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2 if is_active or is_hovered else 1, cv2.LINE_AA)

                label = btn["label"]
                if b_type == "brush_size":
                    label = f"{self.BRUSH_SIZES[self.brush_size_idx]}px"

                font_scale = 0.44
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = x1 + (x2 - x1 - tw) // 2
                text_y = y1 + (y2 - y1 + th) // 2

                text_color = (255, 255, 255) if not is_hovered else (0, 255, 255)
                cv2.putText(
                    frame, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA
                )

        # Temporary Notification Banner
        if self.notification_msg and (time.time() - self.notification_time < 2.5):
            banner_text = self.notification_msg
            (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx1 = (self.width - bw) // 2 - 20
            by1 = self.height - 75
            bx2 = (self.width + bw) // 2 + 20
            by2 = by1 + bh + 22

            banner_overlay = frame.copy()
            cv2.rectangle(banner_overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
            cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame, banner_text, (bx1 + 20, by1 + bh + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
            )

    def blend_with_canvas(self, frame: np.ndarray) -> np.ndarray:
        """Blends the virtual canvas onto the camera frame with vivid color preservation."""
        if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
            self.resize_canvas(frame.shape[1], frame.shape[0])

        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, stroke_mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(stroke_mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        drawing_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=stroke_mask)

        return cv2.add(frame_bg, drawing_fg)
