"""Virtual Calculator Module.

Provides a gesture-controlled virtual calculator with safe AST-based expression evaluation,
visual button feedback, debounce protection against accidental multi-presses,
and support for basic arithmetic operations (+, -, *, /), parentheses, backspace, and clear.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import ast
import operator
import time
import cv2
import numpy as np


class SafeEvaluator(ast.NodeVisitor):
    """Safely evaluates mathematical expressions without unrestricted eval().

    Only permits constants (numbers), binary operators (+, -, *, /, %),
    and unary operators (+, -). Strictly forbids function calls, attribute access,
    variable lookups, and builtins.
    """

    # Allowed binary operators mapping
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
    }

    # Allowed unary operators mapping
    UNARY_OPERATORS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def evaluate(self, expr_str: str) -> Union[int, float]:
        """Parses and evaluates an arithmetic expression safely.

        Args:
            expr_str: Math expression string (e.g., '14 * (3 + 2)').

        Returns:
            Computed numerical value.

        Raises:
            ValueError: If expression contains disallowed operations or syntax.
            ZeroDivisionError: If division by zero occurs.
        """
        # Clean standard display operators
        cleaned = expr_str.replace("×", "*").replace("÷", "/")

        try:
            tree = ast.parse(cleaned, mode="eval")
        except SyntaxError as exc:
            raise ValueError("Invalid syntax") from exc

        return self.visit(tree.body)

    def visit(self, node: ast.AST) -> Union[int, float]:
        """Custom visitor for supported AST nodes."""
        if isinstance(node, ast.Constant):  # Python 3.8+ numeric constant
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

        # Disallow everything else (calls, names, attributes, imports, etc.)
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
        # Active press feedback lasts 0.18s
        if self.is_pressed and (now - self.press_time > 0.18):
            self.is_pressed = False

        x1, y1 = self.x, self.y
        x2, y2 = self.x + self.width, self.y + self.height

        # Determine color styling
        if self.is_pressed:
            fill_color = (0, 180, 255)       # Bright amber/cyan on press
            border_color = (255, 255, 255)
            border_thickness = 3
        elif self.is_hovered:
            fill_color = (80, 80, 80)         # Lighter gray on hover
            border_color = (0, 220, 255)      # Cyan highlight border
            border_thickness = 2
        else:
            fill_color = self.bg_color
            border_color = (85, 85, 85)
            border_thickness = 1

        # Draw button body
        cv2.rectangle(frame, (x1, y1), (x2, y2), fill_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness, cv2.LINE_AA)

        # Draw key label centered
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
    """Full-featured gesture-controlled virtual calculator."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.expression: str = ""
        self.result: str = ""
        self.evaluator = SafeEvaluator()

        # Feedback state
        self.feedback_msg: str = ""
        self.feedback_time: float = 0.0

        # Debounce and pinch tracking
        self.was_pinched: bool = False
        self.last_press_time: float = 0.0
        self.cooldown_sec: float = 0.32  # Minimum seconds between button clicks

        # Dwell hover click tracking
        self.hovered_key: Optional[str] = None
        self.hover_start_time: float = 0.0
        self.dwell_threshold: float = 0.9  # Seconds hovering to trigger automatic click

        # Calculator layout definition
        self.buttons: List[CalculatorButton] = []
        self._init_calculator_layout()

    def resize_layout(self, new_width: int, new_height: int) -> None:
        """Dynamically adjusts layout when frame resolution changes."""
        if (new_width, new_height) != (self.frame_width, self.frame_height):
            self.frame_width = new_width
            self.frame_height = new_height
            self._init_calculator_layout()

    def _init_calculator_layout(self) -> None:
        """Positions calculator panel and buttons centered on the right half of the frame."""
        # Calculator panel dimensions
        panel_w = 460
        panel_h = 560
        start_x = self.frame_width - panel_w - 40
        start_y = (self.frame_height - panel_h) // 2 + 30

        self.panel_rect = (start_x, start_y, start_x + panel_w, start_y + panel_h)
        self.display_rect = (start_x + 15, start_y + 15, start_x + panel_w - 15, start_y + 115)

        # 5 rows x 4 columns layout
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

                # Color accents for operators and special keys
                if key in ["C", "DEL"]:
                    bg = (40, 40, 80)     # Muted reddish
                elif key in ["+", "-", "*", "/", "%", "="]:
                    bg = (70, 50, 30)     # Warm operator color
                elif key in ["(", ")"]:
                    bg = (55, 55, 60)
                else:
                    bg = (45, 45, 45)     # Digits

                self.buttons.append(CalculatorButton(key, bx, by, btn_w, btn_h, bg_color=bg))

    def evaluate_expression(self, is_final: bool = False) -> None:
        """Evaluates current expression safely and stores formatted result.

        Args:
            is_final: If True (e.g. '=' was pressed), errors are explicitly displayed.
                      If False (while typing), incomplete expressions do not display as errors.
        """
        if not self.expression:
            self.result = ""
            return

        try:
            value = self.evaluator.evaluate(self.expression)
            # Format nicely: integer or clean float
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
        """Processes a button press and updates calculator state.

        Args:
            key: The key pressed ('0'-'9', '+', '-', '*', '/', '=', 'C', 'DEL', '.', '(', ')')
        """
        if key == "C":
            self.expression = ""
            self.result = ""
            self.feedback_msg = "Cleared"
        elif key == "DEL":
            if self.expression:
                self.expression = self.expression[:-1]
                self.evaluate_expression(is_final=False)
            self.feedback_msg = "Backspace"
        elif key == "=":
            self.evaluate_expression(is_final=True)
            if self.result and not self.result.startswith("Error"):
                self.expression = self.result
                self.feedback_msg = "Calculated"
        else:
            # If recovering from an error, reset expression on new input
            if self.result.startswith("Error"):
                self.expression = ""
                self.result = ""

            # Length limit to fit display comfortably
            if len(self.expression) < 28:
                # Disallow consecutive invalid operator sequences
                operators = ["+", "-", "*", "/", "%"]
                if key in operators and self.expression and self.expression[-1] in operators:
                    self.expression = self.expression[:-1] + key
                else:
                    self.expression += key

                # Compute live preview if complete
                self.evaluate_expression(is_final=False)

        self.feedback_time = time.time()

    def process_hand(
        self,
        frame: np.ndarray,
        index_tip: Optional[Tuple[int, int]],
        is_pinch: bool = False,
    ) -> Optional[str]:
        """Tracks hand cursor over calculator buttons and triggers presses via pinch or dwell.

        Args:
            frame: Video frame for rendering feedback.
            index_tip: (x, y) coordinates of index fingertip.
            is_pinch: Whether thumb and index are pinched together.

        Returns:
            Key label if a button was triggered this frame, else None.
        """
        if index_tip is None:
            self.was_pinched = False
            self.hovered_key = None
            for btn in self.buttons:
                btn.is_hovered = False
            return None

        x, y = index_tip
        now = time.time()
        triggered_key: Optional[str] = None
        current_hover: Optional[CalculatorButton] = None

        # Reset hover state for all buttons
        for btn in self.buttons:
            if btn.contains(x, y):
                btn.is_hovered = True
                current_hover = btn
            else:
                btn.is_hovered = False

        # Visual cursor indicator
        cursor_color = (0, 220, 255) if is_pinch else (255, 200, 0)
        cv2.circle(frame, (x, y), 9, cursor_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 12, (255, 255, 255), 2, cv2.LINE_AA)

        # Trigger logic 1: Pinch click (fires on rising edge of pinch with cooldown)
        if is_pinch and not self.was_pinched and (now - self.last_press_time > self.cooldown_sec):
            if current_hover is not None:
                triggered_key = current_hover.key
                current_hover.is_pressed = True
                current_hover.press_time = now
                self.last_press_time = now
                self.handle_key(triggered_key)

        # Trigger logic 2: Dwell hover (if hovering steadily over a button for > dwell_threshold)
        if current_hover is not None and not is_pinch:
            if self.hovered_key == current_hover.key:
                dwell_elapsed = now - self.hover_start_time
                progress_ratio = min(1.0, dwell_elapsed / self.dwell_threshold)
                if progress_ratio > 0.15:
                    angle = int(progress_ratio * 360)
                    cv2.ellipse(
                        frame, (x, y), (18, 18), 0, -90, -90 + angle,
                        (0, 255, 120), 3, cv2.LINE_AA
                    )

                if dwell_elapsed >= self.dwell_threshold and (now - self.last_press_time > self.cooldown_sec):
                    triggered_key = current_hover.key
                    current_hover.is_pressed = True
                    current_hover.press_time = now
                    self.last_press_time = now
                    self.hover_start_time = now + 0.3
                    self.handle_key(triggered_key)
            else:
                self.hovered_key = current_hover.key
                self.hover_start_time = now
        else:
            self.hovered_key = None

        self.was_pinched = is_pinch
        return triggered_key

    def draw_ui(self, frame: np.ndarray) -> None:
        """Renders calculator panel, display screen, buttons, and instructions.

        Args:
            frame: Video frame to render onto.
        """
        x1, y1, x2, y2 = self.panel_rect

        # Draw semi-transparent calculator panel background
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
