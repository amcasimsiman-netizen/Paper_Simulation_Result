# Farmer-Harris Cell Model Simulation
Python implementation of the Farmer & Harris discrete cell model studying cell migration, deformation, and clustering due to chemotaxis.

## Requirements
* Python 3.13 (64-bit)
* NumPy
* Matplotlib
* FFmpeg (for MP4 animation rendering)

## Simulation Case Overview
* **`CASE_1.py`**: Single cell migrating toward one signal cell.
* **`CASE_2.py`**: Symmetric arrangement of cells around a signal source.
* **`CASE_3.py`**: Multiple cells aggregating toward a signal source.
* **`CASE_4.py`**: Multiple cells aggregating toward two signal sources.
* **`CASE_5.py`**: Collective migration toward multiple signal cells with time-delayed signaling.

## Running the Cases
Run any self-contained case script to execute the simulation and generate its `.mp4` animation:
```bash
python CASE_1.py
python CASE_2.py
python CASE_3.py
python CASE_4.py
python CASE_5.py
