# LAGO Workshop – ARTI & MEIGA Simulation Environment

This repository contains the material used in the **LAGO Workshop**, focused on cosmic ray simulations with **ARTI** and detector response modeling with **MEIGA**.

---

## 📂 Repository Structure

- **arti.txt**  
  Step-by-step instructions for running ARTI simulations.

- **docker_install.txt**  
  Guide for installing Docker and setting up the containerized ARTI environment.

- **EAS3d.py**  
  Script to visualize air shower development in 3D.

- **EASsec.py**  
  Script to process secondary particles from ARTI outputs.

- **EAStime.py**  
  Script to analyze the time distribution of secondary particles.

- **999900.sec**  
  Example ARTI output file with secondary particles.

- **LAGO Workshop (3).pdf**  
  Slides used during the workshop.

---

## 🚀 Getting Started

### 1. Install Docker
Follow the instructions in [`docker_install.txt`](docker_install.txt) to install Docker and configure your environment.

### 2. Run ARTI in a container
Use the guide in [`arti.txt`](arti.txt) to:
- Pull the ARTI container.
- Mount a local folder for input/output.
- Run simulations of cosmic-ray showers.

### 3. Process Simulation Data
The provided Python scripts allow you to analyze and visualize the ARTI outputs:

```bash
python3 EAS3d.py   # 3D visualization of showers
python3 EASsec.py  # Process secondary particle data
python3 EAStime.py # Analyze time distribution
