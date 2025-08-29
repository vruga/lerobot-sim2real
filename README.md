# LeRobot Sim2Real

LeRobot Sim2real provides code to train with Reinforcement Learning in fast GPU parallelized simulation and rendering via [ManiSkill](https://github.com/haosulab/ManiSkill) and deploy to the real-world. The codebase is designed for use with the [🤗 LeRobot](https://github.com/huggingface/lerobot) library, which handles all of the hardware interfacing code. Once you clone and follow the installation instructions you can try out the [zero-shot RGB sim2real tutorial](./docs/zero_shot_rgb_sim2real.md) to train in pure simulation something that can pick up cubes in the real world like below:

![](./docs/assets/sim2real-demo.gif)

Note that this project is still in a very early stage. There are many ways the sim2real can be improved (like more system ID tools, better reward functions etc.), but we plan to keep this repo extremely simple for readability and hackability.

If you find this project useful, give this repo and [ManiSkill](https://github.com/haosulab/ManiSkill) a star! If you are using [SO100](https://github.com/TheRobotStudio/SO-ARM100/)/[LeRobot](https://github.com/huggingface/lerobot), make sure to also give them a star. If you use ManiSkill / this sim2real codebase in your research, please cite our [research paper](https://arxiv.org/abs/2410.00425):

```bibtex
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Viswesh Nagaswamy Rajesh and Yong Woo Choi and Yen-Ru Chen and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
  journal = {Robotics: Science and Systems},
  year={2025},
}
```

## Getting Started

### Prerequisites
- Python 3.11 (recommended)
- NVIDIA GPU with CUDA support

### Installation with UV Package Manager

We use [UV](https://docs.astral.sh/uv/) for fast and reliable dependency management. UV provides deterministic builds and excellent performance.

#### 1. Install UV
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. Clone and Setup Repository
```bash
# Clone the repository
git clone https://github.com/vruga/lerobot-sim2real.git
cd lerobot-sim2real

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

#### 3. Install PyTorch
```bash
# Install PyTorch - choose the version appropriate for your CUDA version
uv add torch 
```

#### 4. Install LeRobot with OAK Camera Support
We use a maintained fork of LeRobot that includes OAK camera integration and other enhancements:

```bash
# Install our LeRobot fork with OAK camera support
uv add git+https://github.com/vruga/lerobot.git@pr-1363

# This branch includes:
# - OAK-D/OAK-D Pro camera drivers
# - Enhanced depth perception capabilities  
# - Improved camera calibration tools
# - Multi-camera synchronization support
```

> **Note:** This fork (branch `pr-1363`) includes native OAK camera drivers, enhanced calibration tools, and improved hardware compatibility. The OAK integration provides better depth perception and more robust real-world performance. If you're using standard USB cameras, you can still use the original LeRobot repository.

### Verification

#### Test ManiSkill Installation
The ManiSkill/SAPIEN simulator requires working NVIDIA drivers and Vulkan packages. Verify the installation:

```bash
python -m mani_skill.examples.demo_random_action
```

If you encounter issues with drivers/Vulkan, follow the troubleshooting guide: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting

#### Test LeRobot Hardware Connection
```bash
# Test robot connection (adjust for your hardware)
python -c "from lerobot import LeRobotEnv; print('LeRobot installed successfully!')"
```

### Development Setup

For development and contributing:

```bash
# Clone in development mode
git clone https://github.com/vruga/lerobot-sim2real.git
cd lerobot-sim2real

# Install in development mode with all optional dependencies
uv sync --all-extras --dev

```

## Hardware Support

### Supported Robots
- **SO100 Robot Arm** (primary support)
- **SO101 Robot Arm** (experimental - known sim2real differences)

### Camera Support
- **OAK-D/OAK-D Pro** (via our LeRobot fork at `vruga/lerobot@pr-1363`)
- **OpenCV cameras** 
- **RealSense Cameras**

### Getting Your LeRobot Fork

If you want to use your own LeRobot fork with custom modifications:

1. **Fork LeRobot**: Go to https://github.com/huggingface/lerobot and click "Fork"

2. **Add your modifications**: Clone your fork and add features like OAK camera support

3. **Update pyproject.toml**: Add your fork as a dependency:
   ```toml
   [project]
   dependencies = [
       "lerobot @ git+https://github.com/vruga/lerobot.git@pr-1363",
       # ... other dependencies
   ]
   ```

4. **Install**: Run `uv sync` to install your custom fork

**Current Fork Features (pr-1363):**
- Native OAK camera integration with `depthai` support
- Enhanced calibration workflows for depth cameras
- Improved multi-camera synchronization
- Better depth-based manipulation policies

## Quick Start Tutorials

### 1. Zero-Shot RGB Sim2Real (Recommended First)
Train a PPO agent in simulation and deploy zero-shot to real hardware:
- **Tutorial**: [Zero-Shot RGB Sim2Real Guide](./docs/zero_shot_rgb_sim2real.md)
- **Time**: ~2-3 hours training on RTX 4090
- **Hardware**: SO100 + any RGB camera

### 2. Flow Policy Optimization (FPO) 
Advanced training with improved sample efficiency and stability:
- **Tutorial**: [FPO Sim2Real Guide](./docs/fpo_sim2real.md)
- **Time**: ~2-3 hours training on RTX 4090
- **Hardware**: SO100 + RGB camera


#### LeRobot Hardware Issues
- **Robot not detected**: Check USB connections and permissions
- **OAK camera not working**: Ensure `depthai` is properly installed (`uv add depthai`)
- **Standard camera issues**: Install camera-specific drivers
- **Calibration problems**: Use the enhanced calibration tools in the `pr-1363` branch
- **Depth perception issues**: Verify OAK camera depth stream is working



## Citation

If you use this codebase in your research, please cite both ManiSkill and this repository:

```bibtex
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Viswesh Nagaswamy Rajesh and Yong Woo Choi and Yen-Ru Chen and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
  journal = {Robotics: Science and Systems},
  year={2025},
}

@software{lerobot_sim2real,
  title={LeRobot Sim2Real: Fast GPU Simulation to Real World Transfer},
  author={Your Name and Contributors},
  url={https://github.com/StoneT2000/lerobot-sim2real},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **ManiSkill Team**: For the excellent simulation framework
- **HuggingFace LeRobot**: For the hardware interface and robotics toolkit  
- **Community Contributors**: Everyone who has contributed code, documentation, and feedback
- **Hardware Partners**: SO100/SO101 robot arm manufacturers and camera vendors