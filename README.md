# OpenCV Projects

A collection of three computer-vision projects built with Python and OpenCV, covering classical contour analysis, deep-learning-based multi-object tracking, and feature-based image stitching (with an FPGA port of the ORB feature extractor).

| # | Project | Folder | Core techniques |
|---|---------|--------|-----------------|
| 1 | Geometric Shape Recognition & Pose Estimation | [`Objects Detection/`](./Objects%20Detection) | Thresholding, morphology, Canny, contours, image moments, minAreaRect, approxPolyDP |
| 2 | Vehicle Tracking & Collision Warning | [`YOLO Tracker/`](./YOLO%20Tracker) | YOLO, ByteTrack, KDTree path mapping, trajectory extrapolation |
| 3 | Image Stitching via Homography | [`Homography/`](./Homography) | ORB / SIFT, KNN matching + ratio test, RANSAC homography, perspective warp, FPGA (HLS) |

---

## 1. Geometric Shape Recognition & Pose Estimation

Detects the planar geometric objects in an image and reports, for each one, its **shape**, its **(X, Y) position** in a reference coordinate frame drawn in the image, and its **rotation angle**.

**Pipeline**

1. **Pre-processing** – grayscale → binary threshold → morphological closing (9×9) → Gaussian blur → Canny edges → external contours.
2. **Contour split by area** – small contours of a fixed size are the three colour markers that define the coordinate frame (origin `O`, `X` axis, `Y` axis); larger contours are the target objects.
3. **Coordinate frame** – marker roles are identified by the colour at each marker's centroid. The pixel centroid of every object is converted to frame coordinates by solving the 2×2 linear system spanned by the `OX` and `OY` basis vectors, so the result is correct even when the frame is rotated or skewed in the image.
4. **Shape classification** – uses geometric ratios of each contour: aspect ratio of the minimum-area rectangle, and the fill ratio (bounding-box area ÷ contour area). Classes: `square`, `rectangle`, `circle`, `semicircle`, `triangle`.
5. **Rotation angle** – the contour is simplified with `approxPolyDP`; the longest vertex-to-vertex segment gives a principal direction, whose signed angle to the frame's X-axis is computed from the cross and dot products. Shape-specific rules then resolve symmetry and direction ambiguity (90° periodicity for squares, the diagonal offset for rectangles, which side the centroid lies on for triangles and semicircles). Output range: (−180°, 180°].

**Run**

```bash
cd "Objects Detection"
python main.py images/im01.png
```

Prints the number of objects, then one dict per object, e.g. `{'label': 'triangle', 'X': 42, 'Y': 17, 'angle': -35}`.

---

## 2. Vehicle Tracking & Collision Warning

Detects and tracks vehicles in traffic videos, estimates their speed and short-term trajectory, and **warns of a collision before it happens**.

**Pipeline**

```
Video frame → YOLO detection → ByteTrack (track IDs) → KDTree path mapping → speed & trajectory prediction → collision risk
```

1. **Detection** – a custom-trained YOLO model (`yolo-model-weights/best.pt`, via Ultralytics) with confidence 0.25. Outputs are converted with `supervision`.
2. **Multi-object tracking** – `ByteTrackTracker` (Kalman prediction + IoU association) gives every vehicle a persistent track ID.
3. **Path mapping** – the pixel coordinates of the predefined lanes are loaded into a SciPy `KDTree`. Each vehicle centre is matched to its nearest path point (accepted if within 25 px), and a per-ID vote picks the vehicle's most likely path.
4. **Speed & prediction** – the last 21 centre positions per ID are kept in a deque. The average velocity (px/frame) is estimated from samples spaced 4 frames apart and extrapolated to 5 future positions.
5. **Collision risk** – the predicted trajectories of every pair of vehicles are compared at matching future time steps. A conflict is flagged when the distance between two predicted centres is less than half the sum of their box sizes. The sooner the conflict, the higher the risk score (0–100%).

**Results** – on the 40-video test set: **38/40 correct (95%)**, with 19/20 collision videos and 19/20 normal videos classified correctly.
- One false negative: a single vehicle was split into two detections just before impact.
- One false positive: a camera angle that made two passing cars look like they were touching.

**Run**

```bash
cd "YOLO Tracker/CSI 5342-Project"
pip install ultralytics supervision trackers scipy opencv-python
python main.py   # processes every video in Project_Input_Videos/ → Output_Video/
```

See `Assignment-Project(Group8).ipynb` for a step-by-step walkthrough and `Vehicle Tracking and Collision Warning.pptx` for the slides.

---

## 3. Image Stitching via Homography (ORB / SIFT + FPGA)

Stitches two overlapping photos into one larger image.

**Pipeline**

1. **Keypoints & descriptors** – ORB (`merge__orb.py`) or SIFT (`merge__sift.py`).
2. **Matching** – brute-force KNN matching (k = 2) followed by Lowe's ratio test (0.75 for ORB, 0.8 for SIFT).
3. **Homography** – `cv.findHomography` with RANSAC (reprojection threshold 4.0). The SIFT version can reject a pair with too few inliers (`MIN_MATCH_COUNT`).
4. **Warp & blend** – image 1 is warped into image 2's plane with `warpPerspective`, and image 2 is filled into the canvas. `draw_compare()` draws the top-N matches for inspection.

**Run**

```bash
cd Homography
python merge__sift.py   # edit the image paths in __main__ first
python merge__orb.py    # expects foto1A.jpg / foto1B.jpg in the same folder
```

### FPGA acceleration of ORB (HLS)

The ORB feature extractor was also implemented as a hardware IP core with Vitis HLS → [**wangyichengsh/HLS_Project**](https://github.com/wangyichengsh/HLS_Project).

- **What the IP does** – Gaussian blur → FAST corners + non-maximum suppression → intensity-centroid orientation → 256-bit rotated BRIEF (sampling pattern compatible with OpenCV).
- **System** – Kintex-7 (`xc7k420t`). The IP takes an AXI4-Stream input, writes results through AXI4 masters, and talks to a Raspberry Pi 5 host over PCIe via XDMA.
- **Performance** – a 128×128 frame takes about 68.5 µs at 100 MHz (6,847 cycles).
- **Verification**
  - HLS C-simulation: stitches images with the hardware keypoints and compares the result against the OpenCV pipeline.
  - Vivado behavioural simulation: covers the full PCIe → DMA → IP data path.

---

## Requirements

- Python 3.9+
- `opencv-python` (SIFT is included in OpenCV ≥ 4.4), `numpy`
- Project 2 also needs: `ultralytics`, `supervision`, `trackers`, `scipy`
