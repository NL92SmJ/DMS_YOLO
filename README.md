# Deep Dive: Systems-Level Real-Time Telemetry & Hardware Benchmarking – Smart Cockpit DMS & Object Detection

## 1. Motivation & Problem Statement
Modern computer vision applications, real-time video streaming, and advanced driver assistance systems (ADAS) frequently hit severe hardware bottlenecks when scaling beyond naive prototype implementations. While standard computer vision code often runs in isolated environments with generous resource allocations, production-ready Driver Monitoring Systems (DMS) require continuous, deterministic telemetry to ensure driver safety, sub-millisecond reaction times, and highly efficient resource utilization on edge hardware.

This project explores the architectural design, performance characteristics, and hardware footprint of a production-ready **Smart Cockpit Driver Monitoring System (DMS)** built on Ubuntu and Python. Operating on a constrained consumer system—an **NVIDIA RTX 3050 (6 GB VRAM)** paired with an Intel Core i3-12100F—this benchmark investigates how simultaneous multi-model execution—combining real-time object detection (**YOLO ONNX**), landmark estimation (**MediaPipe Face Mesh**), 3D head pose estimation (`solvePnP`), fatigue metrics calculation (EAR/MAR), hardware telemetry logging (`PyNVML`), and synchronous CSV logging—impacts system resources and frame-pacing consistency.

Crucially, the entire pipeline relies strictly on **Edge Computing**: No sensitive video streams leave the device, ensuring full compliance with strict privacy standards (such as GDPR/DSGVO) without sacrificing performance.

> **⚠️ Important Notice / Disclaimer:** 
> This project is a **private hobby project** and serves as a technical Proof of Concept (PoC) to explore edge AI and real-time telemetry. It is explicitly *not* a real driving experiment or an official product. (Although the author works in the automotive industry, this project has no connection whatsoever to professional activities or corporate projects).

---

## 2. Test Environment & Hardware Setup
* **CPU:** Intel Core i3-12100F (4 cores / 8 threads)
* **System RAM:** 16 GB DDR4
* **GPU:** NVIDIA GeForce RTX 3050 — 6 GB VRAM
* **Camera:** Legacy USB 2.0 Webcam (~10 years old, serving as a hardware constraint test for legacy integration)
* **Software Stack:** Python, OpenCV, Ultralytics YOLO (ONNX runtime), MediaPipe Face Mesh, PyNVML, Pandas, Matplotlib

### Test Bench Architecture & Data Flow
```text
[Legacy USB 2.0 Webcam] 
       │ (Asynchronous Capture Thread)
       ▼
[OpenCV / VRAM Pipeline] ──► [YOLO ONNX Inference (RTX 3050)] ──► Object Detection (Smartphone/Person)
       │
       ├──► [MediaPipe Face Mesh (468 Landmarks)] ──► [solvePnP (3D Head Pose: Yaw/Pitch/Roll)]
       │                                                      │
       │                                                      ▼
       │                                             [EMA Filter (α=0.3)]
       │                                                      │
       ▼                                                      ▼
[PyNVML Hardware Polling] ──────────────────────► [State Estimation & CSV Logging]
(GPU Load, VRAM, Power Draw)                            │
                                                        ▼
                                           [Live HUD / Post-Analysis Dashboards]
```

<img width="739" height="507" alt="Bildschirmfoto vom 2026-11-17 09-13-52" src="https://github.com/user-attachments/assets/052e0f26-3b34-478b-81c1-53d9161b618f" />


---

## 3. Deep-Dive Insight: Technical Architecture & Signal Processing
To achieve production-grade reliability and avoid I/O-blocking, the architecture implements a multi-threaded asynchronous video stream (`VideoCaptureAsync`) coupled with a multi-stage inference and filtering pipeline:

### A. Dual-Model Inference Pipeline (YOLO & MediaPipe)
* **YOLO Object Detection (ONNX Model Integration):** 
  * *Model Source & Rationale:* The object detection relies on a customized nano architecture (`yolo26n`), sourced via HuggingFace and adapted for edge deployment. 
  * *Why ONNX?* To decouple the inference engine from heavy Python/PyTorch overhead, the model was converted and optimized into the open standard **ONNX (Open Neural Network Exchange)** format. This ensures platform-independent, highly optimized graph execution and efficient memory layout when running on local GPU runtimes.
* **3D Head Pose Estimation via `solvePnP`:** Maps 468 3D facial landmarks from MediaPipe Face Mesh to a generic 3D head model using OpenCV's Perspective-n-Point algorithm. Decomposing the rotation matrix via `cv2.RQDecomp3x3` extracts precise Euler angles (Yaw, Pitch, Roll).

### B. Signal Stabilization via Exponential Moving Average (EMA)
Raw landmark coordinates naturally suffer from high-frequency jitter. To prevent false triggers in safety-critical states, an Exponential Moving Average (EMA) filter with a smoothing factor (`alpha=0.3`) is applied directly onto the Euler angles:

`Smooth_current = α * Raw_current + (1 - α) * Smooth_previous`

### C. Fatigue Metrics & Discrete State Machine
* **Eye Aspect Ratio (EAR):** Continuously monitors eye openness (`EAR < 0.22`) to track potential micro-sleeps with precise duration counters.
* **Mouth Aspect Ratio (MAR):** Monitors vertical-to-horizontal mouth opening ratios to detect yawns (`MAR > 0.45`).
* **Real-Time Hardware Telemetry via PyNVML:** Polls GPU utilization, VRAM consumption (MB), and power draw (Watts) directly from the NVIDIA RTX 3050 per frame.

---

## 4. Practical Applications & Real-World Use Cases
While developed as an advanced portfolio project, the underlying architecture mirrors industrial-grade edge implementations across several safety-critical domains:
1. **Commercial Fleet Management & Logistics:** Combatting driver fatigue, distracted driving, and smartphone usage on long-haul routes.
2. **Autonomous & Semi-Autonomous Vehicles (Level 2/3):** Verifying driver attentiveness as a mandatory safety layer for safe handovers.
3. **Cross-Industry Transferability (e.g., Medical & Healthcare / AAL):** The core principles of local edge-AI monitoring—such as fall detection using pose estimation in elderly care—solve major industry bottlenecks by providing automated safety triggers while preserving complete patient data privacy (zero cloud upload).
4. **Industrial Heavy Machinery & Aviation:** Localized edge-AI monitoring where operator drowsiness poses catastrophic risks.

---

## 5. Comprehensive Benchmark Results & Telemetry Analysis
Empirical data captured over continuous execution runs demonstrates the lightweight efficiency and high throughput of the architecture on consumer hardware:

| Metric | Performance / Footprint (RTX 3050) |
| :--- | :--- |
| **Average Framerate** | **~57.0 FPS** (High-throughput async pipeline) |
| **Average GPU Utilization** | ~39.6% |
| **Average VRAM Footprint** | ~1,009 MB |
| **Average Power Draw** | ~38.0 W |
| **Head Pose States Tracked** | Geradeaus, Links, Rechts, Oben, Unten |
| **Events Captured** | EAR/MAR Thresholding + YOLO Object/Person Detection Logging |

### Automated Analytics & Visualization
The project includes an automated post-session analysis script (`analyze_dms.py`) that generates high-resolution evaluation assets from CSV logs:


* **`_tech_dashboard.png`:** Multi-panel time-series visualization aligning EAR/MAR, FPS/GPU load, VRAM, and Power consumption.
<img width="3570" height="3248" alt="smart_cockpit_log_20261117_081338_tech_dashboard" src="https://github.com/user-attachments/assets/4c69e6b0-3129-4a40-879a-be87ce26ece4" />


* **`_safety_analysis_chart.png`:** Scenario analysis translating distraction and inactivity durations into physical distances covered at 130 km/h (~36.1 m/s), offering stark visual impact for safety audits.
<img width="2970" height="1414" alt="smart_cockpit_log_20261117_081338_safety_analysis_chart" src="https://github.com/user-attachments/assets/ffee1d08-0d85-4722-907e-226f874e9d4c" />

## 6. Screenshots of Live-Demo
1.) Base-Case:
<img width="1920" height="983" alt="normal" src="https://github.com/user-attachments/assets/7913eb91-7c0d-4a0d-8103-34c6d0b28e37" />

2.) Closed Eyes:
<img width="1920" height="1080" alt="eyes_closed" src="https://github.com/user-attachments/assets/b96e2c87-fb3a-4959-8e47-5a5afb24f259" />

3.) Cellphone:
<img width="1920" height="1080" alt="cellphone" src="https://github.com/user-attachments/assets/c10c4bb2-218b-48e9-a3d9-d3219e3f6c23" />


## 7. Outlook: Hardware Scaling, Edge Advantages & Jetson Workflow

### Why Local GPU Power in Cars Matters (Edge AI Rationale)
1. **Deterministic Latency & Safety:** At 130 km/h (~36 m/s), every millisecond counts. Cloud-based solutions suffer from cellular dead zones and latency fluctuations. Local edge acceleration guarantees deterministic inference times for instant warnings.
2. **Privacy by Design (GDPR):** Cabin monitoring captures sensitive biometric data. Local processing ensures that no video streams leave the vehicle—raw frames are discarded immediately.
3. **Energy Efficiency (TOPS/Watt):** Automotive SoCs are specifically optimized for an ideal ratio of computing performance to strict thermal and power constraints.

### Future Porting Workflow: NVIDIA Jetson Integration
To transition from desktop development (RTX 3050) to an embedded automotive target system (e.g., NVIDIA Jetson Orin Nano / Orin NX), the following steps are planned:
* **Buy an NVIDIA Jetson Developer Kit:** You need to start somewhere ;)
* **JetPack SDK & TensorRT Compilation:** Translating ONNX models into highly optimized native TensorRT `.engine` files to maximize utilization of Tensor Cores.
* **Quantization (FP16/INT8):** Leveraging mixed precision to dramatically reduce memory bandwidth and inference latency.
* **Zero-Copy Camera Pipelines:** Utilizing GStreamer or V4L2 pipelines to handle image capture and scaling directly within GPU VRAM, completely bypassing CPU bottlenecks.
