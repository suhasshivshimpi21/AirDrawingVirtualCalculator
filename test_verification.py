"""Verification test suite for Air Drawing and Virtual Calculator with History Tape."""

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
    assert evaluator.evaluate("15 * 4") == 60, "Multiplication failed"
    assert evaluator.evaluate("120 / 3") == 40.0, "Division failed"
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
    print("Testing VirtualCalculator and Calculation History Tape...")
    calc = VirtualCalculator(frame_width=1536, frame_height=864)

    # 1. Test standard calculation 15 * 4 = 60
    calc.handle_key("1")
    calc.handle_key("5")
    calc.handle_key("*")
    calc.handle_key("4")
    assert calc.expression == "15*4"
    calc.handle_key("=")
    assert calc.expression == "60"

    # Verify equation was recorded into History Tape
    assert len(calc.history) == 1, "Calculation was not added to History Tape"
    assert calc.history[0]["expr"] == "15*4"
    assert calc.history[0]["result"] == "60"
    print("Verified calculation added to History Tape: 15*4 = 60")

    # 2. Test second calculation 120 / 3 = 40
    calc.handle_key("C")
    calc.handle_key("1")
    calc.handle_key("2")
    calc.handle_key("0")
    calc.handle_key("/")
    calc.handle_key("3")
    calc.handle_key("=")
    assert calc.expression == "40"
    assert len(calc.history) == 2
    assert calc.history[1]["result"] == "40"
    print("Verified second calculation in History Tape: 120/3 = 40")

    # 3. Test recalling previous result from History Tape
    # Recall item 0 (which has result '60')
    calc.handle_key("C")
    assert calc.expression == ""
    calc.load_history_result(0)
    assert calc.expression == "60", f"Expected expression '60', got '{calc.expression}'"
    print("Verified History Recall: clicking old result loaded '60' into calculator!")

    # 4. Test Clear History Tape
    calc.clear_history()
    assert len(calc.history) == 0, "History tape was not cleared"
    print("Verified History Tape cleared.")

    # 5. Test button hover and pinch
    mock_frame = np.zeros((864, 1536, 3), dtype=np.uint8)
    calc.draw_ui(mock_frame)

    btn_0 = calc.buttons[0]
    cursor = (btn_0.x + 10, btn_0.y + 10)
    key_pressed = calc.process_hand(mock_frame, cursor, is_pinch=True)
    assert key_pressed == btn_0.key, f"Expected {btn_0.key}, got {key_pressed}"

    print("VirtualCalculator tests passed successfully.")


def test_air_drawer():
    print("Testing AirDrawer and audio integration...")
    drawer = AirDrawer(width=1536, height=864, output_dir="output")
    mock_frame = np.zeros((864, 1536, 3), dtype=np.uint8)

    # Test Drawing mode
    status = drawer.process_hand(mock_frame, (300, 300), [0, 1, 0, 0, 0], is_pinch=False)
    assert status == "DRAWING"
    assert drawer.was_drawing is True
    status2 = drawer.process_hand(mock_frame, (320, 320), [0, 1, 0, 0, 0], is_pinch=False)
    assert status2 == "DRAWING"
    assert np.any(drawer.canvas > 0), "Canvas should have drawn pixels"

    # Test Pause mode
    status_pause = drawer.process_hand(mock_frame, (400, 400), [0, 1, 1, 0, 0], is_pinch=False)
    assert status_pause == "PAUSED"
    assert drawer.was_drawing is False

    # Test Blending & Clear
    blended = drawer.blend_with_canvas(mock_frame)
    assert blended.shape == (864, 1536, 3)
    drawer.clear_canvas()
    assert np.all(drawer.canvas == 0), "Canvas should be completely cleared"

    print("AirDrawer tests passed successfully.")


def test_hand_tracker():
    print("Testing HandTracker initialization...")
    tracker = HandTracker(mode=False, max_hands=1)
    mock_frame = np.zeros((864, 1536, 3), dtype=np.uint8)

    processed = tracker.find_hands(mock_frame, draw=False)
    assert processed.shape == (864, 1536, 3)
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
