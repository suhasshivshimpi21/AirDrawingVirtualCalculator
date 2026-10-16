# Air Drawing and Virtual Calculator

A high-performance Computer Vision desktop application built with **Python 3.11**, **OpenCV**, and **MediaPipe**. This application allows users to draw in mid-air using fingertip tracking and interact with a virtual calculator via hand gestures without touching a mouse or keyboard.

---

## Features

### 1. 📜 Calculation History Tape
- **Recent Calculations Sidebar:** An on-screen history tape sidebar displays your last 3–5 calculated equations (e.g., `15 * 4 = 60`, `120 / 3 = 40`).
- **Interactive Result Recall:** Point and pinch (or hover-dwell) on any previous equation to instantly reload its answer back into your active expression.
- **Clear Tape:** Includes a hands-free `[CLEAR TAPE]` button to reset the calculation log.

### 2. 🔊 Tactile Audio Feedback (Sound Effects)
- **Asynchronous, Non-Blocking Audio:** Powered by Windows `winsound` in background threads for zero frame-rate stutter.
- **Mechanical Button Clicks:** High-pitch acoustic click on every keypad button press.
- **Drawing Chirp:** Subtle tone when drawing begins as you extend your index finger.
- **Calculation Chime:** Harmonious dual-tone chime when an expression is evaluated.
- **Clear Tone:** Low sweep tone when clearing the canvas, history tape, or calculator display.

### 3. 🖥️ True Edge-to-Edge Fullscreen & Auto-Crop
- **Full Display Integration:** Automatically detects native monitor resolution (e.g. 1536x864) and launches in true edge-to-edge fullscreen.
- **Hardware Pillarbox Auto-Crop:** Automatically detects and crops inactive black side borders from 4:3 webcam sensors, delivering an edge-to-edge 16:9 widescreen experience with zero black bars.
- **Instant Fullscreen Toggle:** Press **`F`** or click the on-screen **`[F: FULLSCREEN]`** button.

### 4. 🖐️ High-Accuracy Hand & Gesture Tracking
- **MediaPipe Hands (`model_complexity=1`):** Full landmark model for pinpoint fingertip accuracy and tracking stability.
- **Rotation-Resilient Finger Detection:** Geometric distance invariant calculations ensure pointing and gesture detection work reliably even when the hand is tilted.
- **Adaptive Pinch Scaling:** Automatically adjusts pinch distance threshold according to your distance from the camera.
- **Smooth Continuous Lines:** Real-time exponential moving average filter prevents jitter and produces clean strokes.

### 5. 🎨 Air Drawing Mode
- **7 Vibrant Colors:** Cyan, Green, Red, Orange, Magenta, Yellow, White.
- **4 Brush Sizes:** 4px, 8px, 14px, 22px.
- **Eraser Mode:** Dedicated virtual eraser to clear specific strokes.
- **Pause & Navigate Gesture:** Raise Index + Middle fingers together to move the cursor freely without drawing.
- **Save Snapshot:** Save drawings as PNG images to `output/air_drawing_<timestamp>.png`.

### 6. 🧮 Gesture-Controlled Virtual Calculator
- **Full Keypad:** Digits `0`–`9`, decimal point `.`, arithmetic operations `+`, `-`, `*`, `/`, parentheses `(`, `)`, backspace `DEL`, clear `C`, and evaluate `=`.
- **Hands-Free Button Selection:**
  - **Pinch Click:** Point index finger over any button and pinch thumb + index tip to immediately click.
  - **Hover Dwell:** Hover your index fingertip steadily over any button for 0.85 seconds to trigger a click automatically with a circular progress indicator.
- **Debounce & Anti-Ghosting:** Rising-edge pinch trigger and cooldown timers prevent accidental double-clicks.
- **Safe Math Evaluation:** Custom AST visitor strictly limits operations to basic arithmetic and blocks arbitrary code execution.

---

## Hand Gestures Guide

| Gesture | Mode | Action |
| :--- | :--- | :--- |
| **Index Up** (Middle down) | Air Drawing | **Draws continuous stroke** with active color & brush size |
| **Index + Middle Up** | Air Drawing | **Hover / Pause:** Moves cursor without drawing; hover over toolbar |
| **Pinch (Thumb + Index)** | Air Drawing | **Clicks toolbar buttons** (Color, Clear, Save, Eraser, Size) |
| **Pinch (Thumb + Index)** | Calculator | **Clicks calculator button** or **recalls history item** |
| **Hover Dwell** (0.85s) | Calculator | **Auto-triggers button** with circular progress feedback |
| **Pinch on Top Bar** | Both | **Switches modes** or toggles Fullscreen / Exit |

---

## Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| `F` | **Toggle Fullscreen** / Windowed mode |
| `1` or `D` | Switch to **Air Drawing** mode |
| `2` or `C` | Switch to **Virtual Calculator** mode |
| `Tab` or `M` | **Toggle** between Air Drawing and Virtual Calculator |
| `X` | **Clear** entire drawing canvas |
| `V` | **Cycle** colors |
| `B` | **Cycle** brush sizes (4px, 8px, 14px, 22px) |
| `E` | **Toggle** Eraser mode |
| `S` | **Save** drawing snapshot to `output/` |
| `Q` or `Esc` | **Exit** the application cleanly |

---

## How to Run

Run from your project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe main.py
```

Optional flags:
- `--camera <index>`: Specify webcam device index (default: `0`).
- `--windowed`: Launch in 1280x720 windowed mode instead of fullscreen.
