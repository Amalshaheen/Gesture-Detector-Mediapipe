# Gesture Recognition Pipeline (Camera Stream -> Command Send)

This document explains the actual implementation in `main.py` from live camera frames to sending a final gesture command over Bluetooth, including the mathematical model used at each stage.

## 1) Runtime Components and Data Flow

The pipeline in `main.py` can be viewed as:

1. Camera frame capture (`cv2.VideoCapture`)
2. Frame conversion (`BGR -> RGB`) and MediaPipe image wrapping
3. Asynchronous hand landmark inference (`HandLandmarker.detect_async`)
4. Landmark-to-gesture scoring (`detect_joystick_gesture`)
5. Temporal stabilization (`stabilize_command`)
6. Command transmission (`send_command`) to ESP32 over serial Bluetooth
7. Safety and side channels (emergency Telegram alerts, STOP fallback)

High-level mapping:

$$
I_t \xrightarrow{\text{MediaPipe}} L_t \xrightarrow{\text{gesture score}} g_t^{raw}
\xrightarrow{\text{history filter}} g_t^{stable} \xrightarrow{\text{map}} c_t \xrightarrow{\text{serial}} \text{ESP32}
$$

Where:

- $I_t$: frame at time $t$
- $L_t$: 21 hand landmarks from MediaPipe
- $g_t^{raw}$: raw per-frame gesture label
- $g_t^{stable}$: temporally stabilized gesture label
- $c_t$: single-character command (`F`, `B`, `L`, `R`, `S`, `H`, `E`)

## 2) Camera Streaming and Inference Loop

In the main loop:

1. Read frame:

```python
success, image = cap.read()
```

2. If capture fails, force safety STOP:

```python
reset_command_stabilizer()
send_command("STOP")
handle_emergency_alerts("STOP")
continue
```

3. Convert color and wrap into MediaPipe image:

```python
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
```

4. Trigger async inference with timestamp:

```python
landmarker.detect_async(mp_image, int(time.time() * 1000))
```

5. The callback `print_result` stores the last inference output in `latest_result`.

This creates a producer/consumer pattern: camera frames continue, and the most recent recognized landmarks are consumed in the same loop iteration.

## 3) Landmark Representation

For one hand, MediaPipe returns 21 landmarks:

- Wrist: index 0
- Thumb: 1..4
- Index: 5..8
- Middle: 9..12
- Ring: 13..16
- Little: 17..20

Each landmark has $(x,y,z)$ in normalized image coordinates (with depth-like $z$).

The implementation builds a 2D palm center:

$$
P = \frac{1}{5}\left(L_0 + L_5 + L_9 + L_{13} + L_{17}\right)
$$

using only $x,y$ for several shape metrics.

## 4) Geometric Primitives Used for Scoring

### 4.1 2D distance

The helper `_distance_2d(a, b)` uses:

$$
d(a,b) = \sqrt{(a_x-b_x)^2 + (a_y-b_y)^2}
$$

### 4.2 Joint angle at a middle point

The helper `_joint_angle_deg(a, b, c)` computes angle at $b$:

$$
\vec{u}=a-b,\quad \vec{v}=c-b
$$

$$
\cos\theta = \frac{\vec{u}\cdot\vec{v}}{\|\vec{u}\|\|\vec{v}\|},\quad
\theta = \cos^{-1}(\text{clamp}(\cos\theta,-1,1))
$$

and returns $\theta$ in degrees.

### 4.3 Clamp

Several scores are normalized with:

$$
\text{clamp}(x;0,1)=\max(0,\min(1,x))
$$

## 5) Finger Extension Model

The classifier does not use only a binary open/closed check. It computes continuous extension scores in $[0,1]$.

### 5.1 Non-thumb fingers

For index/middle/ring/little, `_finger_extension_score(...)` blends:

1. Straightness from PIP and DIP angles
2. Reach ratio relative to palm center

Let:

- $\alpha_{pip}$ = angle at PIP
- $\alpha_{dip}$ = angle at DIP
- $r = d(\text{tip},P) / d(\text{mcp},P)$

Then:

$$
s_{straight}=0.5\cdot \text{clamp}\left(\frac{\alpha_{pip}-115}{55}\right)
+0.5\cdot \text{clamp}\left(\frac{\alpha_{dip}-115}{55}\right)
$$

$$
s_{reach}=\text{clamp}\left(\frac{r-0.95}{0.60}\right)
$$

$$
s_{finger}=0.65\cdot s_{straight}+0.35\cdot s_{reach}
$$

### 5.2 Thumb

The thumb uses a separate kinematic chain in `_thumb_extension_score(...)`:

- MCP-like angle from landmarks (1,2,3)
- IP angle from landmarks (2,3,4)
- Reach ratio from tip 4 to palm center

with analogous normalization:

$$
s_{thumb}=0.65\cdot s_{straight}^{thumb}+0.35\cdot s_{reach}^{thumb}
$$

This improves robustness because thumb articulation differs from other fingers.

## 6) Palm Orientation Estimation

The code distinguishes gestures with identical finger shape but opposite palm orientation (for example `HORN` vs `EMERGENCY`, `STOP` vs `BACKWARD`).

### 6.1 Palm normal from 3D landmarks

Using wrist $W=L_0$, index MCP $I=L_5$, little MCP $K=L_{17}$:

$$
\vec{v_1}=I-W,\quad \vec{v_2}=K-W
$$

Palm normal:

$$
\vec{n}=\vec{v_1}\times\vec{v_2}
$$

The implementation uses normalized $n_z$ and adjusts sign by handedness label:

$$
p_{raw}=\frac{n_z}{\|\vec{n}\|}\cdot h
$$

where $h=-1$ for left hand, $h=+1$ for right hand.

### 6.2 Orientation sign calibration

The system auto-calibrates `orientation_sign` when a confident full-open hand is seen. After calibration:

$$
p = p_{raw}\cdot \text{orientation\_sign}
$$

This stabilizes the meaning of palm up/down across user/camera mirroring differences.

### 6.3 Orientation quality score

For expected `DOWN`:

$$
q_{orient}=\text{clamp}\left(\frac{p-0.02}{0.20}\right)
$$

For expected `UP`:

$$
q_{orient}=\text{clamp}\left(\frac{-p-0.02}{0.20}\right)
$$

## 7) Gesture Template Scoring

`detect_joystick_gesture(...)` defines templates for:

- `STOP`, `BACKWARD`, `FORWARD`, `LEFT`, `RIGHT`, `HORN`, `EMERGENCY`

Each template includes:

- Target finger scores (0 folded, 1 extended)
- Expected orientation (`UP` or `DOWN`)
- Weights (`shape_weight`, `orientation_weight`, optional `split_weight`)

### 7.1 Shape quality

Given target vector $T_f$ and measured finger scores $S_f$:

$$
e_{shape}=\frac{1}{N}\sum_f |S_f-T_f|
$$

$$
q_{shape}=1-e_{shape}
$$

### 7.2 Base total score

$$
score = w_s\cdot q_{shape} + w_o\cdot q_{orient}
$$

### 7.3 Split terms for ambiguous gestures

For `HORN`/`EMERGENCY`, a split term favors index+middle being active while thumb/ring/little are passive:

$$
q_{split}=\text{clamp}\left(\frac{\overline{S}_{index,middle}-\overline{S}_{thumb,ring,little}+0.12}{0.95}\right)
$$

plus pair-balance and thumb-match refinements. The final add-on is weighted by `split_weight`.

For `FORWARD`/`LEFT`/`RIGHT`, split emphasizes a single active finger versus others:

$$
q_{split}=\text{clamp}\left(\frac{S_{active}-\overline{S}_{passive}+0.15}{1.0}\right)
$$

then added as:

$$
score \leftarrow score + w_{split}\cdot q_{split}
$$

### 7.4 Decision logic

After sorting all template scores:

1. Reject if best score below confidence threshold:

$$
score_{best} < 0.56 \Rightarrow \text{UNKNOWN}
$$

2. Reject if top-2 margin is too small:

$$
score_{best}-score_{2nd} < 0.06 \Rightarrow \text{UNKNOWN}
$$

3. Reject if orientation quality weak:

$$
q_{orient,best} < 0.45 \Rightarrow \text{UNKNOWN}
$$

4. Extra safety for `STOP` shape quality:

$$
\text{if } g=\text{STOP and } q_{shape}<0.72 \Rightarrow \text{UNKNOWN}
$$

So per-frame output is either one of valid gestures or `UNKNOWN`.

## 8) Temporal Stabilization (Noise Rejection)

Raw frame-wise predictions can flicker. `stabilize_command(raw_command)` applies a history-based filter.

State:

- `COMMAND_HISTORY`: deque, maxlen 8
- `last_stable_command`
- `last_stable_time`

Rules:

1. If raw is `UNKNOWN`:
   - Keep previous stable non-STOP command only for hold window (`COMMAND_HOLD_SECONDS = 0.25`)
   - Otherwise output `STOP`

2. For known labels:
   - Append to history
   - Find most frequent candidate in history

3. Acceptance thresholds:
   - `EMERGENCY` requires count >= 6
   - Other non-STOP commands require count >= 3
   - Otherwise fallback to held command (if still within hold window), else `STOP`

This is effectively a mode filter with class-specific minimum support.

## 9) Command Mapping and Send Logic

Stable gesture keys are mapped by `GESTURE_COMMANDS`:

- `FORWARD -> F`
- `BACKWARD -> B`
- `LEFT -> L`
- `RIGHT -> R`
- `STOP -> S`
- `HORN -> H`
- `EMERGENCY -> E`

`send_command(command_key)` then:

1. Converts key to character
2. Sends only if character changed from last sent command
3. Applies a 0.1 second debounce on command changes
4. Writes UTF-8 byte to serial if Bluetooth connected
5. Otherwise prints demo output and updates internal send state

So transmission condition is:

$$
c_t \neq c_{last} \;\land\; (t-t_{last})\ge 0.1\,s
$$

## 10) Bluetooth Link Setup

`connect_bluetooth()` attempts:

```python
serial.Serial('/dev/rfcomm0', 115200, timeout=1)
```

If successful, command bytes are sent to ESP32. If not, program continues in demo mode (important for debugging recognition independent of hardware).

## 11) Final Stage: Gesture on ESP32

In `hardwarecode.ino`, ESP32 reads incoming chars via `SerialBT.read()` and executes:

- `F`: both motors forward
- `B`: both motors backward
- `L`: turn left
- `R`: turn right
- `S`: stop
- `E`: emergency stop (same motor stop behavior)
- `H`: horn buzzer pulse

There is an additional ultrasonic distance safety override that blocks some forward/turn movement when obstacle distance is below threshold.

## 12) Safety-Critical Behavior Summary

The implementation is safety-biased by design:

1. Camera read failure -> immediate `STOP`
2. No tracked hand -> immediate `STOP`
3. Ambiguous recognition (`UNKNOWN`) -> mostly `STOP` (short hold only)
4. High threshold for `EMERGENCY` activation (history count 6)
5. Orientation checks prevent mirror confusions for same-finger-shape gestures

## 13) End-to-End Example Trace

Example for a `FORWARD` gesture:

1. User shows index-only, palm-down hand.
2. MediaPipe produces landmarks $L_t$.
3. Finger scores become roughly: thumb 0, index 1, middle 0, ring 0, little 0.
4. `FORWARD` template gets highest score and passes confidence/margin/orientation checks.
5. Raw `FORWARD` appears repeatedly in history and crosses non-stop frame threshold (>=3).
6. Stabilizer emits `FORWARD` as stable command.
7. Mapper converts to `F`.
8. `send_command` writes byte `b'F'` to Bluetooth serial.
9. ESP32 receives `F`, sets motor directions/speeds for forward motion.

That is the complete camera-to-command chain in your current code.
