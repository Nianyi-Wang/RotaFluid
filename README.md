# RotaFluid
Physics-InspiredGatedAttentionwithLocal-GlobalModelingforRotatingFluid Simulation

## Dependencies

- PyTorch 1.13.0+cu116 (`pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116`)
- Open3D 0.17.0 (`pip install open3d==0.17.0`)
- SPlisHSPlasH 2.4.0 (for generating training data and fluid particle sampling, https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)
- Tensorpack DataFlow (for reading data, `pip install --upgrade git+https://github.com/tensorpack/dataflow.git`)
- python-prctl (needed by Tensorpack DataFlow; depends on libcap-dev, install with `apt install libcap-dev`)
- msgpack (`pip install msgpack`)
- msgpack-numpy (`pip install msgpack-numpy`)
- python-zstandard (`pip install zstandard` https://github.com/indygreg/python-zstandard)
- partio (https://github.com/wdas/partio)
- SciPy
- OpenVDB with python binding (optional for creating surface meshes, https://github.com/AcademySoftwareFoundation/openvdb)
- plyfile (optional for creating surface meshes, `pip install plyfile`)
- others (`pip install -r requirements.txt`)

The versions match the configuration that we have tested on a system with Ubuntu 18.04.
SPlisHSPlasH 2.4.0 is required for generating training data (ensure that it is compiled in *Release* mode).
We recommend to use the latest versions for all other packages.

## Datasets

We provide two pre-generated datasets for training and evaluating fluid simulation models under rotational conditions. Both datasets are available in a unified Google Drive folder. Refer to the corresponding dataset documentation for detailed generation protocols and statistics.

🔗 Download Link (Google Drive)[link](https://drive.google.com/drive/folders/1FccwWzdbtzyCura_-upAZfoIuwrBgJrX?usp=sharing)

### 1. Augmented Liquid3D Dataset
We enhance the public Liquid3D dataset by introducing continuous boundary rotation. Using the high-precision DF-SPH solver (particle radius $r = 0.025$ m), static container configurations are transformed into dynamically rotating systems. This dataset provides a benchmark for learning fluid behavior under non-inertial reference frames.

### 2. Fueltank-CM Dataset
The **Fuel Tank Continuous Maneuver (Fueltank-CM)** dataset simulates realistic sloshing dynamics in representative aircraft fuel tank geometries. Each sequence begins with a randomly initialized fill ratio and random pitch/roll reorientation to ensure diverse initial conditions. The tank then undergoes a continuous rotation phase about the roll axis, followed by a stationary settling phase to capture damping behavior and equilibrium convergence. This systematic integration of complex internal structures with continuous rotational excitation offers a challenging benchmark for training neural simulators on industrial rotating systems.

### Dataset Generation Scripts (Optional)

If you prefer to generate the datasets locally, scripts are provided in the `Rotate_Dataset` subfolder.

1. Set the path to the `DynamicBoundarySimulator` of SPlisHSPlasH in `datasets/splishsplash_config.py`.
2. Run the desired script from within the `datasets` folder.

```bash
# Rotating the box (gravity fixed)
sh datasets/create_fuel_yemian_rotatebox.sh

# Rotating gravity (box fixed)
sh datasets/create_fuel_yemian_rotategravity.sh

---

## Training the Network

### Training Scripts

RotaFluid supports two training modes: standard fluid dynamics and rotational fluid dynamics with PIGA.

#### Standard Training (without rotation)
```bash
# Standard training
scripts/train_network_torch.py --cfg=scripts/default.yaml

# Rotational training with PIGA
scripts/train_network_torch_rota_optimized.py --cfg=scripts/default_rota.yaml --enable_rotation
```

## Evaluation

```bash
scripts/evaluate_network.py --trainscript=scripts/train_network_torch_rota_optimized.py \
                           --cfg=scripts/default_rota.yaml \
                           --weights=model_weights.pt
```

## Inference

```bash
scripts/run_network.py --weights=model_weights.pt \
                      --scene=scenes/example_scene.json \
                      --output=output_dir \
                      --num_steps=500
```

## License

MIT License
