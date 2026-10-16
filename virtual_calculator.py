"""Virtual Calculator Module.

Provides a gesture-controlled virtual calculator with safe AST-based expression evaluation,
visual button feedback, debounce protection against accidental multi-presses,
calculation history tape sidebar, and interactive sound effects.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import ast
import operator
import time
import cv2
import numpy as np

from sound_effects import (
    play_button_click,
    play_calculate_chime,
    play_clear_tone,
    play_history_recall,
)


class SafeEvaluator(ast.NodeVisitor):
    """Safely evaluates mathematical expressions without unrestricted eval().

    Only permits constants (numbers), binary operators (+, -, *, /, %),
    and unary operators (+, -). Strictly forbids function calls, attribute access,
    variable lookups, and builtins.
    """

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
    }

    UNARY_OPERATORS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def evaluate(self, expr_str: str) -> Union[int, float]:
        """Parses and evaluates an arithmetic expression safely."""
        cleaned = expr_str.replace("×", "*").replace("÷", "/")

        try:
            tree = ast.parse(cleaned, mode="eval")
        except SyntaxError as exc:
            raise ValueError("Invalid syntax") from exc

        return self.visit(tree.body)

    def visit(self, node: ast.AST) -> Union[int, float]:
        """Custom visitor for supported AST nodes."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        if isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op_type = type(node.op)

            if op_type in self.OPERATORS:
                if op_type is ast.Div and right == 0:
                    raise ZeroDivisionError("Division by zero")
                return self.OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        if isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            op_type = type(node.op)
            if op_type in self.UNARY_OPERATORS:
                return self.UNARY_OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

        raise ValueError(f"Disallowed expression node: {type(node).__name__}")


class CalculatorButton:
    """Represents a single calculator button on the virtual interface."""

    def __init__(
        self,
        key: str,
        x: int,
        y: int,
        width: int,
        height: int,
        bg_color: Tuple[int, int, int] = (50, 50, 50),
        accent_color: Tuple[int, int, int] = (220, 220, 220),
    ):
        self.key = key
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.accent_color = accent_color

        self.is_hovered: bool = False
        self.is_pressed: bool = False
        self.press_time: float = 0.0

    def contains(self, px: int, py: int) -> bool:
        """Checks if a point (px, py) lies within this button."""
        return self.x <= px <= (self.x + self.width) and self.y <= py <= (self.y + self.height)

    def draw(self, frame: np.ndarray) -> None:
        """Renders the button with dynamic hover and pressed visual states."""
        now = time.time()
        if self.is_pressed and (now - self.press_time > 0.18):
            self.is_pressed = False

        x1, y1 = self.x, self.y
        x2, y2 = self.x + self.width, self.y + self.height

        if self.is_pressed:
            fill_color = (0, 180, 255)
            border_color = (255, 255, 255)
            border_thickness = 3
        elif self.is_hovered:
            fill_color = (80, 80, 80)
            border_color = (0, 220, 255)
            border_thickness = 2
        else:
            fill_color = self.bg_color
            border_color = (85, 85, 85)
            border_thickness = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), fill_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness, cv2.LINE_AA)

        font_scale = 0.85
        thickness = 2
        (tw, th), _ = cv2.getTextSize(self.key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = x1 + (self.width - tw) // 2
        ty = y1 + (self.height + th) // 2

        text_color = (255, 255, 255) if not self.is_hovered else (0, 240, 255)
        if self.is_pressed:
            text_color = (20, 20, 20)

        cv2.putText(
            frame, self.key, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA
        )


class VirtualCalculator:
    """Full-featured gesture-controlled virtual calculator with history tape and sound effects."""

    def __init__(self, frame_width: int = 1536, frame_height: int = 864):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.expression: str = ""
        self.result: str = ""
        self.evaluator = SafeEvaluator()

        # Calculation History Tape (stores last 5 calculations: [{"expr": str, "result": str}])
        self.history: List[Dict[str, str]] = []
        self.hovered_history_idx: Optional[int] = None
        self.pressed_history_idx: Optional[int] = None
        self.pressed_history_time: float = 0.0

        # Feedback state
        self.feedback_msg: str = ""
        self.feedback_time: float = 0.0

        # Debounce and pinch tracking
        self.was_pinched: bool = False
        self.last_press_time: float = 0.0
        self.cooldown_sec: float = 0.30

        # Dwell hover click tracking
        self.hovered_target_id: Optional[str] = None
        self.hover_start_time: float = 0.0
        self.dwell_threshold: float = 0.85

        # Layout geometry
        self.buttons: List[CalculatorButton] = []
        self._init_calculator_layout()

    def resize_layout(self, new_width: int, new_height: int) -> None:
        """Dynamically adjusts layout when frame resolution changes."""
        if (new_width, new_height) != (self.frame_width, self.frame_height):
            self.frame_width = new_width
            self.frame_height = new_height
            self._init_calculator_layout()

    def _init_calculator_layout(self) -> None:
        """Positions calculator keypad on the right and history tape on the left."""
        panel_w = 460
        panel_h = 560
        start_x = self.frame_width - panel_w - 40
        start_y = (self.frame_height - panel_h) // 2 + 30

        self.panel_rect = (start_x, start_y, start_x + panel_w, start_y + panel_h)
        self.display_rect = (start_x + 15, start_y + 15, start_x + panel_w - 15, start_y + 115)

        # History Tape Sidebar layout (left side of frame)
        hist_w = 360
        hist_h = panel_h
        hx1 = max(30, start_x - hist_w - 40)
        hy1 = start_y
        self.history_rect = (hx1, hy1, hx1 + hist_w, hy1 + hist_h)
        self.btn_clear_history_rect = (hx1 + 20, hy1 + hist_h - 48, hx1 + hist_w - 20, hy1 + hist_h - 14)

        # 5 rows x 4 columns keypad
        key_layout = [
            ["C", "(", ")", "/"],
            ["7", "8", "9", "*"],
            ["4", "5", "6", "-"],
            ["1", "2", "3", "+"],
            ["0", ".", "DEL", "="],
        ]

        btn_grid_top = start_y + 130
        btn_w = 95
        btn_h = 70
        spacing_x = 12
        spacing_y = 12

        self.buttons = []
        for r_idx, row in enumerate(key_layout):
            for c_idx, key in enumerate(row):
                bx = start_x + 15 + c_idx * (btn_w + spacing_x)
                by = btn_grid_top + r_idx * (btn_h + spacing_y)

                if key in ["C", "DEL"]:
                    bg = (40, 40, 80)
                elif key in ["+", "-", "*", "/", "%", "="]:
                    bg = (70, 50, 30)
                elif key in ["(", ")"]:
                    bg = (55, 55, 60)
                else:
                    bg = (45, 45, 45)

                self.buttons.append(CalculatorButton(key, bx, by, btn_w, btn_h, bg_color=bg))

    def add_to_history(self, expr_str: str, result_str: str) -> None:
        """Records a successful calculation onto the history tape (keeps last 5)."""
        if not expr_str or not result_str:
            return
        # Avoid duplicate consecutive entries
        if self.history and self.history[-1]["expr"] == expr_str and self.history[-1]["result"] == result_str:
            return

        self.history.append({"expr": expr_str, "result": result_str})
        if len(self.history) > 5:
            self.history = self.history[-5:]

    def load_history_result(self, idx: int) -> None:
        """Loads a recalled calculation result from history tape into active expression."""
        if 0 <= idx < len(self.history):
            recalled_val = self.history[idx]["result"]
            self.expression = str(recalled_val)
            self.result = ""
            self.feedback_msg = f"Recalled: {recalled_val}"
            self.pressed_history_idx = idx
            self.pressed_history_time = time.time()
            play_history_recall()

    def clear_history(self) -> None:
        """Clears calculation history tape."""
        self.history.clear()
        self.feedback_msg = "Tape Cleared"
        play_clear_tone()

    def evaluate_expression(self, is_final: bool = False) -> None:
        """Evaluates current expression safely and stores formatted result."""
        if not self.expression:
            self.result = ""
            return

        try:
            value = self.evaluator.evaluate(self.expression)
            if isinstance(value, float):
                if value.is_integer():
                    self.result = str(int(value))
                else:
                    self.result = f"{value:.6f}".rstrip("0").rstrip(".")
            else:
                self.result = str(value)
        except ZeroDivisionError:
            self.result = "Error: Div by 0"
        except (ValueError, SyntaxError):
            if is_final:
                self.result = "Error: Invalid Expr"
            else:
                self.result = ""
        except Exception:
            if is_final:
                self.result = "Error"
            else:
                self.result = ""

    def handle_key(self, key: str) -> None:
        """Processes a button press, handles audio feedback, and updates calculation state."""
        if key == "C":
            self.expression = ""
            self.result = ""
            self.feedback_msg = "Cleared"
            play_clear_tone()
        elif key == "DEL":
            if self.expression:
                self.expression = self.expression[:-1]
                self.evaluate_expression(is_final=False)
            self.feedback_msg = "Backspace"
            play_button_click()
        elif key == "=":
            prev_expr = self.expression
            self.evaluate_expression(is_final=True)
            if self.result and not self.result.startswith("Error"):
                # Save to history tape and play chime
                self.add_to_history(prev_expr, self.result)
                self.expression = self.result
                self.feedback_msg = "Calculated"
                play_calculate_chime()
            else:
                play_clear_tone()
        else:
            play_button_click()
            if self.result.startswith("Error"):
                self.expression = ""
                self.result = ""

            if len(self.expression) < 28:
                operators = ["+", "-", "*", "/", "%"]
                if key in operators and self.expression and self.expression[-1] in operators:
                    self.expression = self.expression[:-1] + key
                else:
                    self.expression += key

                self.evaluate_expression(is_final=False)

        self.feedback_time = time.time()

    def _get_history_card_rect(self, card_idx: int) -> Tuple[int, int, int, int]:
        """Calculates bounding box for a given history tape entry card."""
        hx1, hy1, hx2, hy2 = self.history_rect
        card_w = (hx2 - hx1) - 30
        card_h = 62
        spacing = 10
        top_offset = hy1 + 75

        cx1 = hx1 + 15
        cy1 = top_offset + card_idx * (card_h + spacing)
        cx2 = cx1 + card_w
        cy2 = cy1 + card_h
        return (cx1, cy1, cx2, cy2)

    def process_hand(
        self,
        frame: np.ndarray,
        index_tip: Optional[Tuple[int, int]],
        is_pinch: bool = False,
    ) -> Optional[str]:
        """Tracks hand cursor over keypad buttons and history cards, triggering clicks via pinch or dwell."""
        if index_tip is None:
            self.was_pinched = False
            self.hovered_target_id = None
            self.hovered_history_idx = None
            for btn in self.buttons:
                btn.is_hovered = False
            return None

        x, y = index_tip
        now = time.time()
        triggered_key: Optional[str] = None
        current_hover_button: Optional[CalculatorButton] = None
        current_hover_history_idx: Optional[int] = None
        is_hovering_clear_history = False

        # 1. Check keypad button hovers
        for btn in self.buttons:
            if btn.contains(x, y):
                btn.is_hovered = True
                current_hover_button = btn
            else:
                btn.is_hovered = False

        # 2. Check history tape item hovers
        self.hovered_history_idx = None
        for i in range(len(self.history)):
            cx1, cy1, cx2, cy2 = self._get_history_card_rect(i)
            if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                self.hovered_history_idx = i
                current_hover_history_idx = i
                break

        # 3. Check Clear History button
        chx1, chy1, chx2, chy2 = self.btn_clear_history_rect
        if chx1 <= x <= chx2 and chy1 <= y <= chy2 and len(self.history) > 0:
            is_hovering_clear_history = True

        # Cursor visualization
        cursor_color = (0, 220, 255) if is_pinch else (255, 200, 0)
        cv2.circle(frame, (x, y), 9, cursor_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 12, (255, 255, 255), 2, cv2.LINE_AA)

        # Determine current hover target identifier
        target_id = None
        if current_hover_button is not None:
            target_id = f"btn_{current_hover_button.key}"
        elif current_hover_history_idx is not None:
            target_id = f"hist_{current_hover_history_idx}"
        elif is_hovering_clear_history:
            target_id = "clear_tape"

        # Click Trigger 1: Pinch Click (rising edge)
        if is_pinch and not self.was_pinched and (now - self.last_press_time > self.cooldown_sec):
            if current_hover_button is not None:
                triggered_key = current_hover_button.key
                current_hover_button.is_pressed = True
                current_hover_button.press_time = now
                self.last_press_time = now
                self.handle_key(triggered_key)
            elif current_hover_history_idx is not None:
                self.load_history_result(current_hover_history_idx)
                self.last_press_time = now
                triggered_key = f"RECALL [{self.history[current_hover_history_idx]['result']}]"
            elif is_hovering_clear_history:
                self.clear_history()
                self.last_press_time = now
                triggered_key = "CLEAR_TAPE"

        # Click Trigger 2: Dwell Hover Click
        if target_id is not None and not is_pinch:
            if self.hovered_target_id == target_id:
                dwell_elapsed = now - self.hover_start_time
                progress_ratio = min(1.0, dwell_elapsed / self.dwell_threshold)

                if progress_ratio > 0.15:
                    angle = int(progress_ratio * 360)
                    cv2.ellipse(
                        frame, (x, y), (18, 18), 0, -90, -90 + angle,
                        (0, 255, 120), 3, cv2.LINE_AA
                    )

                if dwell_elapsed >= self.dwell_threshold and (now - self.last_press_time > self.cooldown_sec):
                    self.last_press_time = now
                    self.hover_start_time = now + 0.35

                    if current_hover_button is not None:
                        triggered_key = current_hover_button.key
                        current_hover_button.is_pressed = True
                        current_hover_button.press_time = now
                        self.handle_key(triggered_key)
                    elif current_hover_history_idx is not None:
                        self.load_history_result(current_hover_history_idx)
                        triggered_key = f"RECALL [{self.history[current_hover_history_idx]['result']}]"
                    elif is_hovering_clear_history:
                        self.clear_history()
                        triggered_key = "CLEAR_TAPE"
            else:
                self.hovered_target_id = target_id
                self.hover_start_time = now
        else:
            self.hovered_target_id = None

        self.was_pinched = is_pinch
        return triggered_key

    def draw_ui(self, frame: np.ndarray) -> None:
        """Renders calculation history tape sidebar, keypad, display screen, and buttons."""
        now = time.time()

        # ----------------- 1. CALCULATION HISTORY TAPE SIDEBAR -----------------
        hx1, hy1, hx2, hy2 = self.history_rect
        hist_overlay = frame.copy()
        cv2.rectangle(hist_overlay, (hx1, hy1), (hx2, hy2), (26, 26, 26), -1)
        cv2.addWeighted(hist_overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (65, 65, 65), 1, cv2.LINE_AA)

        # History Header Title
        cv2.putText(
            frame, "HISTORY TAPE", (hx1 + 18, hy1 + 36),
            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Click any result to load into calculator", (hx1 + 18, hy1 + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA
        )
        cv2.line(frame, (hx1 + 18, hy1 + 68), (hx2 - 18, hy1 + 68), (50, 50, 50), 1)

        # Render History Cards
        if not self.history:
            cv2.putText(
                frame, "No calculations yet.", (hx1 + 25, hy1 + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 120, 120), 1, cv2.LINE_AA
            )
            cv2.putText(
                frame, "Use '=' to record equations here.", (hx1 + 25, hy1 + 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA
            )
        else:
            for idx, item in enumerate(self.history):
                cx1, cy1, cx2, cy2 = self._get_history_card_rect(idx)
                is_hovered = (self.hovered_history_idx == idx)
                is_pressed = (self.pressed_history_idx == idx and (now - self.pressed_history_time < 0.22))

                if is_pressed:
                    card_bg = (0, 180, 255)
                    card_border = (255, 255, 255)
                    border_w = 2
                elif is_hovered:
                    card_bg = (60, 60, 60)
                    card_border = (0, 220, 255)
                    border_w = 2
                else:
                    card_bg = (38, 38, 38)
                    card_border = (65, 65, 65)
                    border_w = 1

                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), card_bg, -1)
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), card_border, border_w, cv2.LINE_AA)

                # Expression line (upper)
                expr_text = f"{item['expr']} ="
                if len(expr_text) > 22:
                    expr_text = "..." + expr_text[-20:]
                cv2.putText(
                    frame, expr_text, (cx1 + 14, cy1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200) if not is_pressed else (20, 20, 20), 1, cv2.LINE_AA
                )

                # Result line (prominent bottom)
                res_text = f"= {item['result']}"
                res_col = (0, 240, 255) if not is_pressed else (10, 10, 10)
                cv2.putText(
                    frame, res_text, (cx1 + 14, cy1 + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, res_col, 2, cv2.LINE_AA
                )

                # Click prompt badge when hovered
                if is_hovered:
                    cv2.putText(
                        frame, "[LOAD]", (cx2 - 65, cy1 + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA
                    )

            # Clear History Tape button at bottom
            chx1, chy1, chx2, chy2 = self.btn_clear_history_rect
            is_ch_hover = (self.hovered_target_id == "clear_tape")
            cv2.rectangle(frame, (chx1, chy1), (chx2, chy2), (55, 35, 35) if not is_ch_hover else (90, 45, 45), -1)
            cv2.rectangle(frame, (chx1, chy1), (chx2, chy2), (120, 60, 60) if not is_ch_hover else (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(
                frame, "CLEAR TAPE", (chx1 + (chx2 - chx1 - 85) // 2, chy1 + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA
            )

        # ----------------- 2. RIGHT-SIDE CALCULATOR KEYPAD -----------------
        x1, y1, x2, y2 = self.panel_rect
        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (x1, y1), (x2, y2), (28, 28, 28), -1)
        cv2.addWeighted(panel_overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 2, cv2.LINE_AA)

        # Display screen
        dx1, dy1, dx2, dy2 = self.display_rect
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (18, 18, 18), -1)
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (90, 90, 90), 1, cv2.LINE_AA)

        expr_display = self.expression if self.expression else "0"
        if len(expr_display) > 22:
            expr_display = "..." + expr_display[-20:]

        cv2.putText(
            frame, expr_display, (dx1 + 15, dy1 + 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA
        )

        if self.result:
            result_display = f"= {self.result}"
            res_color = (0, 230, 255) if not self.result.startswith("Error") else (0, 70, 255)
            cv2.putText(
                frame, result_display, (dx1 + 15, dy1 + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, res_color, 2, cv2.LINE_AA
            )

        for btn in self.buttons:
            btn.draw(frame)

        cv2.putText(
            frame, "GESTURE CALCULATOR", (x1 + 15, y1 - 10),
            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA
        )
