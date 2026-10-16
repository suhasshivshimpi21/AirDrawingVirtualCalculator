"""Verification test suite for Air Drawing and Virtual Calculator modules."""

import os
import numpy as np
import cv2

from hand_tracker import HandTracker
from air_drawing import AirDrawer
from virtual_calculator import SafeEvaluator, VirtualCalculator


def test_safe_evaluator():
    print("Testing SafeEvaluator...")
    evaluator = SafeEvaluator()

    assert evaluator.evaluate("2 + 2") == 4, "2 + 2 failed"
    assert evaluator.evaluate("10 - 4") == 6, "Subtraction failed"
    assert evaluator.evaluate("6 * 7") == 42, "Multiplication failed"
    assert evaluator.evaluate("20 / 4") == 5.0, "Division failed"
    assert evaluator.evaluate("2 + 3 * 4") == 14, "Precedence failed"
    assert evaluator.evaluate("(2 + 3) * 4") == 20, "Parentheses failed"

    # Division by zero
    try:
        evaluator.evaluate("10 / 0")
        assert False, "Should have raised ZeroDivisionError"
    except ZeroDivisionError:
        pass

    print("SafeEvaluator tests passed successfully.")


def test_virtual_calculator():
    print("Testing VirtualCalculator...")
    calc = VirtualCalculator(frame_width=1280, frame_height=720)

    # Test key presses
    calc.handle_key("1")
    calc.handle_key("2")
    calc.handle_key("+")
    calc.handle_key("8")
    assert calc.expression == "12+8"
    assert calc.result == "20"

    calc.handle_key("=")
    assert calc.expression == "20"

    calc.handle_key("DEL")
    assert calc.expression == "2"

    calc.handle_key("C")
    assert calc.expression == ""
    assert calc.result == ""

    # Test button hover and pinch
    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    calc.draw_ui(mock_frame)

    btn_0 = calc.buttons[0]
    cursor = (btn_0.x + 10, btn_0.y + 10)
    key_pressed = calc.process_hand(mock_frame, cursor, is_pinch=True)
    assert key_pressed == btn_0.key, f"Expected {btn_0.key}, got {key_pressed}"

    print("VirtualCalculator tests passed successfully.")


def test_air_drawer():
    print("Testing AirDrawer...")
    drawer = AirDrawer(width=1280, height=720, output_dir="output")
    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Test Drawing mode
    status = drawer.process_hand(mock_frame, (300, 300), [0, 1, 0, 0, 0], is_pinch=False)
    assert status == "DRAWING"
    status2 = drawer.process_hand(mock_frame, (320, 320), [0, 1, 0, 0, 0], is_pinch=False)
    assert status2 == "DRAWING"
    assert np.any(drawer.canvas > 0), "Canvas should have drawn pixels"

    # Test Pause mode
    status_pause = drawer.process_hand(mock_frame, (400, 400), [0, 1, 1, 0, 0], is_pinch=False)
    assert status_pause == "PAUSED"

    # Test Blending & Clear
    blended = drawer.blend_with_canvas(mock_frame)
    assert blended.shape == (720, 1280, 3)
    drawer.clear_canvas()
    assert np.all(drawer.canvas == 0), "Canvas should be completely cleared"

    print("AirDrawer tests passed successfully.")


def test_hand_tracker():
    print("Testing HandTracker initialization...")
    tracker = HandTracker(mode=False, max_hands=1)
    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    processed = tracker.find_hands(mock_frame, draw=False)
    assert processed.shape == (720, 1280, 3)
    lm_list, bbox = tracker.find_positions(mock_frame)
    assert lm_list == []
    assert bbox == (0, 0, 0, 0)
    assert tracker.get_index_tip() is None
    assert tracker.fingers_up() == [0, 0, 0, 0, 0]

    tracker.close()
    print("HandTracker tests passed successfully.")


if __name__ == "__main__":
    test_safe_evaluator()
    test_virtual_calculator()
    test_air_drawer()
    test_hand_tracker()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
