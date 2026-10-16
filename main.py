"""Main Application: Air Drawing and Virtual Calculator.

Combines real-time hand tracking (MediaPipe), virtual air drawing with continuous line smoothing
and color palette controls, and a gesture-driven virtual calculator with safe AST expression evaluation.
Supports true edge-to-edge fullscreen and auto-cropping of camera letterboxing/pillarboxing.
"""

from typing import Tuple, Optional
import argparse
import sys
import time
import cv2
import numpy as np

from hand_tracker import HandTracker
from air_drawing import AirDrawer
from virtual_calculator import VirtualCalculator


def get_screen_resolution() -> Tuple[int, int]:
    """Retrieves primary display monitor resolution on Windows."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1280, 720


class AirDrawingCalculatorApp:
    """Coordinates webcam capture, hand tracking, drawing canvas, and calculator interface."""

    MODE_DRAW = "DRAW"
    MODE_CALC = "CALC"

    def __init__(
        self,
        camera_index: int = 0,
        fullscreen: bool = True,
    ):
        self.camera_index = camera_index
        self.is_fullscreen = fullscreen

        # Detect native monitor screen resolution
        screen_w, screen_h = get_screen_resolution()
        self.width = screen_w if self.is_fullscreen else 1280
        self.height = screen_h if self.is_fullscreen else 720

        # Initialize VideoCapture
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            print(
                f"[WARNING] Camera index {self.camera_index} could not be opened. "
                "Attempting fallback to index 1...",
                file=sys.stderr,
            )
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print(
                    "[ERROR] No accessible webcam found! Ensure your webcam is connected "
                    "and not in use by another app.",
                    file=sys.stderr,
                )

        # Active camera bounds for auto-cropping hardware pillarbox bars
        self.crop_bounds: Optional[Tuple[int, int, int, int]] = None
        self._calibrate_camera_bounds()

        # Application state
        self.mode = self.MODE_DRAW
        self.fps: float = 0.0
        self.prev_frame_time: float = time.time()
        self.window_name = "Air Drawing & Virtual Calculator"

        # Instantiate modular components
        self.tracker = HandTracker(
            mode=False,
            max_hands=1,
            model_complexity=1,     # High-accuracy landmark model
            detection_con=0.55,     # Robust detection in variable room lighting
            track_con=0.55,
        )
        self.drawer = AirDrawer(width=self.width, height=self.height)
        self.calculator = VirtualCalculator(frame_width=self.width, frame_height=self.height)

        # Setup top toolbar navigation coordinates
        self._update_ui_layout()

        # Click cooldown
        self.last_nav_click: float = 0.0
        self.nav_cooldown: float = 0.40

    def _calibrate_camera_bounds(self) -> None:
        """Detects whether webcam driver adds pillarbox/letterbox black bars and computes active crop."""
        if not self.cap or not self.cap.isOpened():
            return

        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            return

        h, w = test_frame.shape[:2]
        col_means = np.mean(test_frame, axis=(0, 2))
        non_zero_cols = np.where(col_means > 1.5)[0]

        if len(non_zero_cols) > 0:
            left_col = int(non_zero_cols[0])
            right_col = int(non_zero_cols[-1]) + 1
            active_w = right_col - left_col

            # If there are significant black borders (> 40px each side)
            if left_col > 40 and (w - right_col) > 40:
                # 16:9 widescreen crop of the active area
                target_h = int(active_w * (self.height / self.width))
                if target_h <= h:
                    top_row = (h - target_h) // 2
                    bottom_row = top_row + target_h
                    self.crop_bounds = (top_row, bottom_row, left_col, right_col)
                else:
                    self.crop_bounds = (0, h, left_col, right_col)
                print(f"[INFO] Auto-detected camera pillarbox bars. Active crop: {self.crop_bounds}")
            else:
                self.crop_bounds = None
        else:
            self.crop_bounds = None

    def _update_ui_layout(self) -> None:
        """Configures button bounds according to current resolution."""
        btn_y1 = 12
        btn_y2 = 60

        # Mode buttons (Left side of top bar)
        self.mode_btn_draw = (18, btn_y1, 155, btn_y2)
        self.mode_btn_calc = (165, btn_y1, 305, btn_y2)

        # Fullscreen & Exit buttons (Right side of top bar)
        self.btn_fullscreen = (self.width - 225, btn_y1, self.width - 110, btn_y2)
        self.btn_exit = (self.width - 100, btn_y1, self.width - 18, btn_y2)

        # Update submodules with current dimensions
        self.drawer.resize_canvas(self.width, self.height)
        self.calculator.resize_layout(self.width, self.height)

    def toggle_fullscreen(self) -> None:
        """Toggles between edge-to-edge fullscreen and windowed view."""
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            screen_w, screen_h = get_screen_resolution()
            self.width, self.height = screen_w, screen_h
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            self.width, self.height = 1280, 720
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)

        self._update_ui_layout()
        self.drawer._notify(f"Fullscreen: {'ON' if self.is_fullscreen else 'OFF'}")

    def switch_mode(self, new_mode: str) -> None:
        """Switches active application mode."""
        if new_mode != self.mode:
            self.mode = new_mode
            self.drawer.previous_point = None
            self.drawer.smoothed_point = None
            mode_name = "Air Drawing" if new_mode == self.MODE_DRAW else "Virtual Calculator"
            self.drawer._notify(f"Switched to {mode_name}")

    def _handle_navigation_gestures(
        self,
        cursor_pt: Optional[Tuple[int, int]],
        is_pinch: bool,
    ) -> bool:
        """Handles top header tab switching and fullscreen/exit gestures."""
        if cursor_pt is None:
            return False

        cx, cy = cursor_pt
        now = time.time()
        if (now - self.last_nav_click) < self.nav_cooldown:
            return False

        # Mode Tab 1: Air Draw
        dx1, dy1, dx2, dy2 = self.mode_btn_draw
        if dx1 <= cx <= dx2 and dy1 <= cy <= dy2 and is_pinch:
            self.last_nav_click = now
            self.switch_mode(self.MODE_DRAW)
            return True

        # Mode Tab 2: Calculator
        cx1, cy1, cx2, cy2 = self.mode_btn_calc
        if cx1 <= cx <= cx2 and cy1 <= cy <= cy2 and is_pinch:
            self.last_nav_click = now
            self.switch_mode(self.MODE_CALC)
            return True

        # Fullscreen Toggle Button
        fx1, fy1, fx2, fy2 = self.btn_fullscreen
        if fx1 <= cx <= fx2 and fy1 <= cy <= fy2 and is_pinch:
            self.last_nav_click = now
            self.toggle_fullscreen()
            return True

        # Quit Button
        qx1, qy1, qx2, qy2 = self.btn_exit
        if qx1 <= cx <= qx2 and qy1 <= cy <= qy2 and is_pinch:
            self.last_nav_click = now
            return "QUIT"

        return False

    def draw_top_navigation_bar(
        self,
        frame: np.ndarray,
        cursor_pt: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Renders top header navigation tabs, fullscreen toggle, and quit buttons."""
        cx = cursor_pt[0] if cursor_pt else -1
        cy = cursor_pt[1] if cursor_pt else -1

        # Semi-transparent top navigation bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, 70), (28, 28, 28), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.line(frame, (0, 70), (self.width, 70), (60, 60, 60), 1)

        # Tab 1: Air Drawing
        dx1, dy1, dx2, dy2 = self.mode_btn_draw
        draw_active = (self.mode == self.MODE_DRAW)
        draw_hover = (dx1 <= cx <= dx2 and dy1 <= cy <= dy2)
        bg_draw = (0, 160, 230) if draw_active else ((85, 85, 85) if draw_hover else (45, 45, 45))
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), bg_draw, -1)
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 255, 255) if draw_active else (100, 100, 100), 2 if draw_active else 1, cv2.LINE_AA)
        cv2.putText(
            frame, "1: AIR DRAW", (dx1 + 16, dy1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2 if draw_active else 1, cv2.LINE_AA
        )

        # Tab 2: Virtual Calculator
        cx1, cy1, cx2, cy2 = self.mode_btn_calc
        calc_active = (self.mode == self.MODE_CALC)
        calc_hover = (cx1 <= cx <= cx2 and cy1 <= cy <= cy2)
        bg_calc = (0, 160, 230) if calc_active else ((85, 85, 85) if calc_hover else (45, 45, 45))
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), bg_calc, -1)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255) if calc_active else (100, 100, 100), 2 if calc_active else 1, cv2.LINE_AA)
        cv2.putText(
            frame, "2: CALCULATOR", (cx1 + 10, cy1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2 if calc_active else 1, cv2.LINE_AA
        )

        # Fullscreen Button (Right side)
        fx1, fy1, fx2, fy2 = self.btn_fullscreen
        f_hover = (fx1 <= cx <= fx2 and fy1 <= cy <= fy2)
        bg_f = (75, 75, 75) if f_hover else (45, 45, 45)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), bg_f, -1)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 220, 255) if f_hover else (90, 90, 90), 1, cv2.LINE_AA)
        fs_label = "F: WINDOWED" if self.is_fullscreen else "F: FULLSCREEN"
        cv2.putText(
            frame, fs_label, (fx1 + 10, fy1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA
        )

        # Quit Button (Right side)
        qx1, qy1, qx2, qy2 = self.btn_exit
        q_hover = (qx1 <= cx <= qx2 and qy1 <= cy <= qy2)
        bg_q = (180, 40, 40) if q_hover else (70, 30, 30)
        cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), bg_q, -1)
        cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), (255, 100, 100) if q_hover else (120, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(
            frame, "Q: QUIT", (qx1 + 14, qy1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA
        )

    def draw_footer_bar(
        self,
        frame: np.ndarray,
        gesture_status: str,
        hand_detected: bool,
    ) -> None:
        """Renders the bottom status bar with shortcuts, FPS, and gesture indicators."""
        footer_h = 36
        y1 = self.height - footer_h
        y2 = self.height

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y1), (self.width, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, y1), (self.width, y1), (50, 50, 50), 1)

        # Left: Hand status and current gesture
        if hand_detected:
            status_text = f"HAND: Active  |  GESTURE: {gesture_status}"
            status_color = (0, 255, 120)
        else:
            status_text = "HAND: Not Detected (Show hand to camera)"
            status_color = (0, 100, 255)

        cv2.putText(
            frame, status_text, (18, y1 + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 1, cv2.LINE_AA
        )

        # Center: Keyboard shortcuts guide
        shortcuts = "Keys: [1] Draw  [2] Calc  [Tab] Mode  [F] Fullscreen  [X] Clear  [V] Color  [S] Save  [Q] Exit"
        cv2.putText(
            frame, shortcuts, (self.width // 2 - 320, y1 + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Right: FPS Counter
        fps_text = f"FPS: {int(self.fps)}"
        cv2.putText(
            frame, fps_text, (self.width - 95, y1 + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA
        )

    def run(self) -> None:
        """Main application execution loop."""
        if not self.cap or not self.cap.isOpened():
            print("[ERROR] Cannot start application because webcam is not opened.", file=sys.stderr)
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.is_fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self.window_name, self.width, self.height)

        print("[INFO] Application running. Press 'Q' or 'Esc' to exit, 'F' for Fullscreen.")

        try:
            while True:
                success, frame = self.cap.read()
                if not success or frame is None:
                    time.sleep(0.02)
                    continue

                # Auto-crop camera hardware pillarbox black bars if present
                if self.crop_bounds is not None:
                    y1_c, y2_c, x1_c, x2_c = self.crop_bounds
                    frame = frame[y1_c:y2_c, x1_c:x2_c]

                # Mirror frame horizontally for intuitive interaction
                frame = cv2.flip(frame, 1)

                # Scale frame cleanly to current screen/window resolution
                if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

                # Track hands
                frame = self.tracker.find_hands(frame, draw=True, custom_style=True)
                lm_list, _ = self.tracker.find_positions(frame)
                hand_detected = len(lm_list) > 0

                index_tip = self.tracker.get_index_tip() if hand_detected else None
                fingers = self.tracker.fingers_up() if hand_detected else [0, 0, 0, 0, 0]
                is_pinch, pinch_dist, pinch_mid = (
                    self.tracker.is_pinch(threshold=None, frame=frame, draw=False)
                    if hand_detected
                    else (False, 999.0, (0, 0))
                )

                # Check top navigation gestures (mode switch, fullscreen, exit)
                nav_action = self._handle_navigation_gestures(index_tip, is_pinch)
                if nav_action == "QUIT":
                    break

                gesture_status = "STANDBY"

                # ----------------- AIR DRAWING MODE -----------------
                if self.mode == self.MODE_DRAW:
                    gesture_status = self.drawer.process_hand(
                        frame=frame,
                        index_tip=index_tip,
                        fingers=fingers,
                        is_pinch=is_pinch,
                    )
                    frame = self.drawer.blend_with_canvas(frame)
                    self.draw_top_navigation_bar(frame, cursor_pt=index_tip)
                    self.drawer.draw_ui(frame, current_point=index_tip)

                # ------------- VIRTUAL CALCULATOR MODE -------------
                elif self.mode == self.MODE_CALC:
                    frame = self.drawer.blend_with_canvas(frame)

                    clicked_key = self.calculator.process_hand(
                        frame=frame,
                        index_tip=index_tip,
                        is_pinch=is_pinch,
                    )
                    if clicked_key:
                        gesture_status = f"CLICKED [{clicked_key}]"
                    elif is_pinch:
                        gesture_status = "PINCH"
                    elif index_tip is not None:
                        gesture_status = "HOVER"
                    else:
                        gesture_status = "STANDBY"

                    self.calculator.draw_ui(frame)
                    self.draw_top_navigation_bar(frame, cursor_pt=index_tip)

                # FPS Calculation
                now = time.time()
                dt = now - self.prev_frame_time
                self.prev_frame_time = now
                if dt > 0:
                    current_fps = 1.0 / dt
                    self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps

                # Render footer status bar
                self.draw_footer_bar(frame, gesture_status, hand_detected)

                # Show frame
                cv2.imshow(self.window_name, frame)

                # Handle keyboard inputs
                key = cv2.waitKey(1) & 0xFF

                # Q or ESC -> Exit
                if key in (ord("q"), ord("Q"), 27):
                    print("[INFO] Clean exit requested by user.")
                    break
                # Fullscreen toggle: F or F11
                elif key in (ord("f"), ord("F")):
                    self.toggle_fullscreen()
                # Mode switching: 1/D or 2/C or Tab/M
                elif key in (ord("1"), ord("d"), ord("D")):
                    self.switch_mode(self.MODE_DRAW)
                elif key in (ord("2"), ord("c"), ord("C")):
                    self.switch_mode(self.MODE_CALC)
                elif key in (ord("\t"), ord("m"), ord("M")):
                    new_mode = self.MODE_CALC if self.mode == self.MODE_DRAW else self.MODE_DRAW
                    self.switch_mode(new_mode)
                # Drawing controls
                elif key in (ord("x"), ord("X")):
                    self.drawer.clear_canvas()
                elif key in (ord("v"), ord("V")):
                    self.drawer.cycle_color()
                elif key in (ord("b"), ord("B")):
                    self.drawer.cycle_brush_size()
                elif key in (ord("e"), ord("E")):
                    self.drawer.toggle_eraser()
                elif key in (ord("s"), ord("S")):
                    saved_path = self.drawer.save_canvas()
                    print(f"[INFO] Snapshot saved: {saved_path}")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Safely closes video stream, MediaPipe solutions, and OpenCV windows."""
        print("[INFO] Releasing system resources...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.tracker.close()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete.")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Air Drawing and Virtual Calculator with MediaPipe and OpenCV."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Launch in windowed mode instead of fullscreen.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the application."""
    args = parse_arguments()
    app = AirDrawingCalculatorApp(
        camera_index=args.camera,
        fullscreen=not args.windowed,
    )
    app.run()


if __name__ == "__main__":
    main()
