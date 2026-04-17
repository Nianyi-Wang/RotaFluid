# Fuel Tank Dataset Generator

A physics-based dataset generation pipeline for fluid dynamics simulation in rotating fuel tanks. Uses [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) to generate training data for machine learning models.

---

## Dataset Analysis

The generated datasets, including complete analysis results, are publicly available at the following Google Drive link:

🔗 Fuel Tank Dataset Analysis & Download：“https://drive.google.com/drive/folders/1FccwWzdbtzyCura_-upAZfoIuwrBgJrX?usp=sharing”

The shared folder contains two primary datasets produced by this pipeline:
1. Augmented Liquid3D Dataset
2. Fueltank-CM Dataset

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Configuration](#configuration)
- [Output Format](#output-format)

---

## Overview

This project generates simulation datasets of fluid behavior inside rotating fuel tanks. The pipeline:

1. Computes rotation sequences (harmonic, constant velocity, triangular profiles)
2. Creates SPlisHSPlasH scene files with rotating tank geometry
3. Runs batch physics simulations across multiple parameter combinations
4. Compresses and packages results into `.zst` archives for ML training

---

## Project Structure

```
fuel_dataset/
├── calculate_rotation_normal.py              # Rotation math utilities
├── compare_rotation.py                       # Validate rotation calculations
│
├── create_physics_fueltank_zh_16_rotate.py           # Scene creator: 16-cell tank
├── create_physics_fueltank_zh_69r_rotate.py          # Scene creator: 69-cell tank
├── create_physics_fueltank_zh_911_rotate.py          # Scene creator: 911-cell tank
├── create_physics_fueltank_zh_rotatebox.py           # Scene creator: simple box
│
├── batch_simulation_rotatebox.py              # Batch runner: box tank
│
├── compress_dataset_rotationsequence_fueltank.py  # Compress simulation output
├── compress_dataset_rotationsequence_2.py         # Compression variant
│
├── batch_process_rotatefueltank.sh            # Main pipeline orchestrator
└── run_batch_simulations_cycle.sh             # Run all simulations + compress
```

---

## Dependencies

### Python Packages

```bash
pip install numpy scipy open3d msgpack msgpack-numpy zstandard tqdm matplotlib
```

| Package | Purpose |
|---|---|
| `numpy` | Array math, rotation matrices |
| `scipy` | Spatial rotations (`scipy.spatial.transform`) |
| `open3d` | Mesh loading and point cloud sampling |
| `msgpack` + `msgpack-numpy` | Binary serialization |
| `zstandard` | `.zst` compression |
| `tqdm` | Progress bars |
| `matplotlib` | Rotation validation plots |

### External Tools

| Tool | Purpose | Notes |
|---|---|---|
| SPlisHSPlasH | Fluid physics simulator | Must be compiled separately |
| Volume Sampling Tool | Convert `.obj` mesh to fluid particles | Bundled with SPlisHSPlasH |

### Model Assets

Tank geometry `.obj` files must be placed in the models directory (see [Setup](#setup)):

- `wallbox.obj` — simulation boundary box
- `tank_16.obj` / `fluid_16.obj` — 16-cell tank geometry
- `tank_69r.obj` / `fluid_69r.obj` — 69-cell tank geometry
- `tank_911.obj` / `fluid_911.obj` — 911-cell tank geometry
- `tank_box.obj` / `fluid_box.obj` — simple box tank

---

## Setup

### 1. Install SPlisHSPlasH

Follow the [official build instructions](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH). After building, note the paths to `SPHSimulator` and `VolumeSampling`.

### 2. Create `splishsplash_config.py`

```python
SIMULATOR_BIN = "/path/to/SPHSimulator"
VOLUME_SAMPLING_BIN = "/path/to/VolumeSampling"
```

### 3. Set up directory structure

The scripts expect the following layout (hardcoded in scripts, adjust as needed):

```
/home/zh/fueltank_datasets/
├── models/
│   └── fueltank/
│       ├── wallbox.obj
│       ├── tank_16.obj
│       ├── fluid_16.obj
│       └── ...
├── output/          # Raw simulation output
├── datasets/        # Processed datasets
└── compressed_datasets/  # Final .zst archives
```

> **Windows users:** The scripts use Linux paths (`/home/zh/...`). You must update all hardcoded paths before running.

### 4. Install Python dependencies

```bash
pip install numpy scipy open3d msgpack msgpack-numpy zstandard tqdm matplotlib
```

---

## Usage

### Run the full pipeline

```bash
bash run_batch_simulations_cycle.sh
```

This runs all batch simulations for all tank variants, then compresses the output.

### Run simulation + inference pipeline

```bash
bash batch_process_rotatefueltank.sh
```

This generates `.zst` compressed datasets, then runs network inference to produce `.npz` files.

### Run a single tank variant

```bash
python batch_simulation_69r_rotate.py
```

### Compress existing simulation output

```bash
python compress_dataset_rotationsequence_fueltank.py
```

### Validate rotation calculations

```bash
python compare_rotation.py
```

---

## Data Pipeline

```
Rotation Parameters
       │
       ▼
calculate_rotation_normal.py
(compute rotation sequences)
       │
       ▼
create_physics_fueltank_zh_*.py
(generate SPlisHSPlasH scene files)
       │
       ▼
SPlisHSPlasH Simulator
(run fluid simulation → raw particle data)
       │
       ▼
compress_dataset_rotationsequence_fueltank.py
(pack to .zst archives)
       │
       ▼
run_network_fueltank_simid.py
(network inference → .npz files)
```

### Rotation Sequence Types

Defined in `calculate_rotation_normal.py`:

| Type | Description |
|---|---|
| Constant velocity | Fixed angular velocity throughout |
| Acceleration/deceleration | Ramp up then ramp down |
| Harmonic | Sinusoidal oscillation |
| Triangular | Linear ramp profile |
| Cycle | Combined multi-phase sequence |

---

## Configuration

Key simulation parameters (set inside each `create_physics_fueltank_zh_*.py`):

| Parameter | Value | Description |
|---|---|---|
| Particle radius | `0.025` | SPH particle size |
| Simulation duration | `6–14 s` | Varies by tank model |
| Export FPS | `50` | Output frame rate |
| Surface tension | `0.2` | Fluid surface tension |
| Viscosity | `0.01` | Fluid viscosity |
| Boundary method | `0` (particle-based) | SPlisHSPlasH boundary handling |

Batch simulation parameters (set inside each `batch_simulation_*.py`):

| Parameter | Values | Description |
|---|---|---|
| Rotation time | `2.0 s` | Duration of each rotation |
| Pitch angles | `30°, 45°, 60°` | Tank tilt angles |
| Repeats per config | `8` | Number of runs per parameter set |

---

## Output Format

### Raw simulation output

SPlisHSPlasH produces per-frame particle data in `.bgeo` format under the `output/` directory, organized by simulation ID.

### Compressed dataset

`compress_dataset_rotationsequence_fueltank.py` packs simulation frames into `.zst` archives using msgpack serialization. Each archive contains:

- Particle positions per frame
- Rotation matrices per frame
- Simulation metadata (angles, timing, tank model)

### Inference output

`run_network_fueltank_simid.py` produces `.npz` files with model predictions alongside ground truth.
