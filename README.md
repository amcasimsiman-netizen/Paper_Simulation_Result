# Farmer-Harris Cell Model Simulation

Python implementation of the Farmer & Harris discrete cell membrane model studying cell migration, deformation, 
and clustering due to chemotaxis.

## Requirements
* Python 3.13 (64-bit)
* NumPy
* Matplotlib
* FFmpeg (for MP4 animation rendering)

## Simulation Case Overview
* **`case1.py`**: Single cell migrating toward one signal cell.
* **`case2.py`**: Symmetric arrangement of cells around a signal source.
* **`case3.py`**: Multiple cells aggregating toward a signal source.
* **`case4.py`**: Multiple cells aggregating toward two signal sources.
* **`case5.py`**: Collective migration toward multiple signal cells with time-delayed signaling.

## Running the Cases
Run any self-contained case script to execute the simulation and generate its `.mp4` animation:
```bash
python case1.py
python case2.py
python case3.py
python case4.py
python case5.py