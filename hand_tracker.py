"""Hand Tracking Module using MediaPipe Hands.

Provides real-time hand detection, landmark extraction, rotation-resilient finger extension tracking,
adaptive pinch detection, and gesture analysis for computer vision applications.
"""

from typing import List, Tuple, Optional, Dict, Any
import math
import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """Detects hands, extracts 21 3D landmarks, and accurately identifies hand gestures.

    Attributes:
        mode (bool): If True, treats input images as batch static images.
        max_hands (int): Maximum number of hands to detect.
        model_complexity (int): Model complexity (1=full for high accuracy).
        detection_con (float): Minimum detection confidence (0.55 for reliable detection in variable lighting).
        track_con (float): Minimum tracking confidence.
    """

    # Landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    TIP_IDS = [4, 8, 12, 16, 20]
    PIP_IDS = [3, 6, 10, 14, 18]
    MCP_IDS = [2, 5, 9, 13, 17]

    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 1,
        model_complexity: int = 1,  # Full model for superior landmark accuracy
        detection_con: float = 0.55,  # Robust detection in various lighting conditions
        track_con: float = 0.55,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.results: Any = None
        self.landmark_list: List[Tuple[int, int, int]] = []  # [(id, x, y)]
        self.hand_type: str = "Right"

    def find_hands(
        self,
        frame: np.ndarray,
        draw: bool = True,
        custom_style: bool = True,
    ) -> np.ndarray:
        """Processes an image frame to detect hands.

        Args:
            frame: BGR image from OpenCV.
            draw: Whether to render landmark points and connections.
            custom_style: Whether to use smooth custom styling.

        Returns:
            The frame with landmarks optionally drawn.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        self.results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if custom_style:
                    self._draw_styled_landmarks(frame, hand_landmarks)
                else:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                    )

        return frame

    def _draw_styled_landmarks(self, frame: np.ndarray, hand_landmarks: Any) -> None:
        """Renders clean, illuminated landmark skeleton and joint nodes."""
        height, width = frame.shape[:2]
        coords = [
            (int(lm.x * width), int(lm.y * height))
            for lm in hand_landmarks.landmark
        ]

        # Draw connecting skeleton lines
        for start_idx, end_idx in self.mp_hands.HAND_CONNECTIONS:
            pt1 = coords[start_idx]
            pt2 = coords[end_idx]
            cv2.line(frame, pt1, pt2, (220, 220, 220), 2, cv2.LINE_AA)

        # Draw joint nodes
        for idx, pt in enumerate(coords):
            if idx in self.TIP_IDS:
                # Fingertips: outer ring and vibrant center
                cv2.circle(frame, pt, 7, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 9, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, (80, 160, 240), -1, cv2.LINE_AA)

    def find_positions(
        self,
        frame: np.ndarray,
        hand_no: int = 0,
    ) -> Tuple[List[Tuple[int, int, int]], Tuple[int, int, int, int]]:
        """Extracts pixel coordinates for all 21 landmarks of the specified hand.

        Args:
            frame: The current image frame.
            hand_no: Index of the detected hand (default 0).

        Returns:
            A tuple of (landmark_list, bounding_box)
            landmark_list: List of (id, x, y) tuples.
            bounding_box: (xmin, ymin, xmax, ymax) enclosing the hand.
        """
        self.landmark_list = []
        bbox = (0, 0, 0, 0)

        if not self.results or not self.results.multi_hand_landmarks:
            return self.landmark_list, bbox

        if hand_no >= len(self.results.multi_hand_landmarks):
            return self.landmark_list, bbox

        # Determine handedness if available
        if self.results.multi_handedness and hand_no < len(self.results.multi_handedness):
            self.hand_type = self.results.multi_handedness[hand_no].classification[0].label

        target_hand = self.results.multi_hand_landmarks[hand_no]
        height, width = frame.shape[:2]

        x_coords: List[int] = []
        y_coords: List[int] = []

        for idx, lm in enumerate(target_hand.landmark):
            cx, cy = int(lm.x * width), int(lm.y * height)
            self.landmark_list.append((idx, cx, cy))
            x_coords.append(cx)
            y_coords.append(cy)

        if x_coords and y_coords:
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)
            bbox = (xmin, ymin, xmax, ymax)

        return self.landmark_list, bbox

    def get_hand_scale(self) -> float:
        """Computes reference hand scale based on palm size (wrist to middle MCP)."""
        if len(self.landmark_list) < 21:
            return 100.0
        w_x, w_y = self.landmark_list[self.WRIST][1], self.landmark_list[self.WRIST][2]
        m_x, m_y = self.landmark_list[self.MIDDLE_MCP][1], self.landmark_list[self.MIDDLE_MCP][2]
        return max(30.0, float(math.hypot(m_x - w_x, m_y - w_y)))

    def fingers_up(self) -> List[int]:
        """Accurately detects which fingers are extended upward or outwards.

        Uses rotation-invariant geometric distance checks combined with vertical
        coordinates, ensuring accurate detection even if the hand is tilted.

        Returns:
            List of 5 integers [thumb, index, middle, ring, pinky] where
            1 represents extended and 0 represents curled/folded.
        """
        if len(self.landmark_list) < 21:
            return [0, 0, 0, 0, 0]

        fingers: List[int] = []
        w_x, w_y = self.landmark_list[self.WRIST][1], self.landmark_list[self.WRIST][2]

        # 1. Thumb detection:
        # Distance from thumb tip to pinky MCP (id 17) or wrist vs thumb IP (id 3)
        t_tip_x, t_tip_y = self.landmark_list[self.THUMB_TIP][1], self.landmark_list[self.THUMB_TIP][2]
        t_ip_x, t_ip_y = self.landmark_list[self.THUMB_IP][1], self.landmark_list[self.THUMB_IP][2]
        p_mcp_x, p_mcp_y = self.landmark_list[self.PINKY_MCP][1], self.landmark_list[self.PINKY_MCP][2]

        dist_tip_pinky = math.hypot(t_tip_x - p_mcp_x, t_tip_y - p_mcp_y)
        dist_ip_pinky = math.hypot(t_ip_x - p_mcp_x, t_ip_y - p_mcp_y)
        dist_tip_wrist = math.hypot(t_tip_x - w_x, t_tip_y - w_y)
        dist_ip_wrist = math.hypot(t_ip_x - w_x, t_ip_y - w_y)

        # Thumb is extended if it's spread out from pinky base and further from wrist than IP
        if (dist_tip_pinky > dist_ip_pinky * 1.15) and (dist_tip_wrist > dist_ip_wrist * 1.05):
            fingers.append(1)
        else:
            fingers.append(0)

        # 2. Four Fingers: Index (8), Middle (12), Ring (16), Pinky (20)
        # Uses dual check: Tip is above PIP (y check) OR distance from wrist to tip is significantly greater than wrist to PIP
        for tip_id, pip_id, mcp_id in zip(self.TIP_IDS[1:], self.PIP_IDS[1:], self.MCP_IDS[1:]):
            tip_x, tip_y = self.landmark_list[tip_id][1], self.landmark_list[tip_id][2]
            pip_x, pip_y = self.landmark_list[pip_id][1], self.landmark_list[pip_id][2]
            mcp_x, mcp_y = self.landmark_list[mcp_id][1], self.landmark_list[mcp_id][2]

            dist_wrist_tip = math.hypot(tip_x - w_x, tip_y - w_y)
            dist_wrist_pip = math.hypot(pip_x - w_x, pip_y - w_y)
            dist_mcp_tip = math.hypot(tip_x - mcp_x, tip_y - mcp_y)
            dist_mcp_pip = math.hypot(pip_x - mcp_x, pip_y - mcp_y)

            # Extended when tip is above PIP or stretched outward from MCP/wrist
            is_above = tip_y < pip_y
            is_extended_dist = (dist_wrist_tip > dist_wrist_pip * 1.10) and (dist_mcp_tip > dist_mcp_pip * 1.15)

            if is_above or is_extended_dist:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def count_total_fingers(self, frame: np.ndarray) -> Tuple[int, List[int], List[Tuple[int, int, int]]]:
        """Counts total extended fingers across all detected hands (supports up to 2 hands).

        Returns:
            Tuple of (total_count, primary_fingers_up, primary_landmarks)
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return 0, [0, 0, 0, 0, 0], []

        total_count = 0
        primary_fingers = [0, 0, 0, 0, 0]
        primary_lm: List[Tuple[int, int, int]] = []

        num_hands = len(self.results.multi_hand_landmarks)
        for h_idx in range(num_hands):
            lm_list, _ = self.find_positions(frame, hand_no=h_idx)
            if lm_list:
                f_up = self.fingers_up()
                total_count += sum(f_up)
                if h_idx == 0:
                    primary_fingers = f_up
                    primary_lm = lm_list

        # Reset internal state to primary hand (index 0)
        if primary_lm:
            self.find_positions(frame, hand_no=0)

        return total_count, primary_fingers, primary_lm

    def find_distance(
        self,
        p1: int,
        p2: int,
        frame: Optional[np.ndarray] = None,
        draw: bool = True,
        color: Tuple[int, int, int] = (255, 0, 255),
    ) -> Tuple[float, Tuple[int, int]]:
        """Calculates Euclidean pixel distance between two landmarks."""
        if len(self.landmark_list) < 21:
            return 9999.0, (0, 0)

        x1, y1 = self.landmark_list[p1][1], self.landmark_list[p1][2]
        x2, y2 = self.landmark_list[p2][1], self.landmark_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)

        if draw and frame is not None:
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x1, y1), 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x2, y2), 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1, cv2.LINE_AA)

        return length, (cx, cy)

    def is_pinch(
        self,
        threshold: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
        draw: bool = False,
    ) -> Tuple[bool, float, Tuple[int, int]]:
        """Determines if the thumb tip and index fingertip are pinched together.

        Dynamically scales the threshold according to hand size so pinching is equally
        reliable whether the user sits close or far from the webcam.
        """
        dist, center = self.find_distance(
            self.THUMB_TIP, self.INDEX_TIP, frame=frame, draw=draw
        )

        if threshold is None:
            # Adaptive threshold based on hand scale (palm height)
            hand_scale = self.get_hand_scale()
            adaptive_thresh = max(26.0, min(65.0, hand_scale * 0.32))
            is_active = dist < adaptive_thresh
        else:
            is_active = dist < threshold

        return is_active, dist, center

    def get_index_tip(self) -> Optional[Tuple[int, int]]:
        """Returns pixel (x, y) coordinates of the index fingertip if detected."""
        if len(self.landmark_list) >= 21:
            return (
                self.landmark_list[self.INDEX_TIP][1],
                self.landmark_list[self.INDEX_TIP][2],
            )
        return None

    def close(self) -> None:
        """Releases MediaPipe Hands resources."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()
