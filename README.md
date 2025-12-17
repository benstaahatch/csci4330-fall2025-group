# MPI + OpenMP Parallel Random Forest

A high-performance implementation of Random Forest using MPI (Message Passing Interface) for distributed computing and OpenMP for shared-memory parallelism. This hybrid approach achieves excellent performance with **99%+ accuracy** and **99%+ F1 scores** on the diabetes prediction dataset.

## docs table of contents etc.

[troubleshooting](docs/troubleshooting.md)

## Features

- **Hybrid Parallelization**: Combines MPI for distributed computing across multiple nodes/ranks with OpenMP for parallel tree building within each rank
- **High Accuracy**: Achieves 99.6% accuracy and 99.4% F1 score
- **Scalable**: Efficiently distributes tree training across MPI ranks
- **Optimized**: Uses histogram-based split finding and binned data representation for fast training

## Project Structure

```
.
├── src/
│   ├── main.cpp              # Hybrid RF main program (MPI + OpenMP)
│   ├── main_mpi_forest.cpp   # MPI Forest main program
│   ├── mpi_forest.cpp        # MPI Random Forest implementation
│   ├── forest.cpp             # Random Forest implementation
│   ├── tree_builder.cpp      # Decision tree builder
│   ├── histogram.cpp         # Histogram-based split finding
│   └── bins.cpp              # Data binning functions
├── include/
│   ├── mpi_forest.h          # MPI Forest header
│   ├── forest.h              # Random Forest header
│   ├── tree_builder.h        # Tree builder header
│   ├── histogram.h           # Histogram header
│   └── bins.h                # Binning header
├── Makefile                  # Build configuration
├── diabetes.csv              # Dataset file
└── README.md                 # This file
```

## Dependencies

- **MPI**: OpenMPI or MPICH
- **OpenMP**: OpenMP library (libomp)
- **C++ Compiler**: C++17 compatible (clang++ or g++)
- **Operating System**: macOS, Linux, or Unix-like system

### Installation (macOS)

```bash
# Install MPI
brew install open-mpi

# Install OpenMP
brew install libomp
```

### Installation (Linux)

```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev openmpi-bin libomp-dev

# CentOS/RHEL
sudo yum install openmpi-devel libomp-devel
```

## Building

### Build MPI Forest

```bash
make mpi_forest
```

### Build Hybrid RF

```bash
make hybrid_rf
```

### Build All

```bash
make
```

## Running

### MPI Forest (Recommended)

Run with 2 MPI ranks:
```bash
mpirun -np 2 ./mpi_forest
```

Run with 4 MPI ranks:
```bash
mpirun -np 4 ./mpi_forest
```

### Hybrid RF

Run with 2 MPI ranks:
```bash
mpirun -np 2 ./hybrid_rf
```

## Model Configuration

The current configuration achieves optimal performance:

- **Number of Trees**: 500
- **Max Depth**: 20
- **Min Samples Split**: 5
- **Max Features**: All features (8)
- **Random Seed**: 42 (for reproducibility)

## Performance

### Expected Results

- **Accuracy**: ~99.6%
- **Precision**: ~100%
- **Recall**: ~98.9%
- **F1 Score**: ~99.4%

### Example Output

```
============================================================
Hybrid MPI + OpenMP Random Forest
============================================================
Loaded dataset: 768 samples, 8 features
Training forest with 500 trees using 2 MPI ranks and OpenMP threads...

===== FULL MODEL METRICS =====
Accuracy:    99.6094%
Precision:   100%
Recall:      98.8806%
Overhead:    0.390625%
F1 Score:    99.4371%
================================
```
## Dataset

The code uses the `diabetes.csv` dataset with:
- **Samples**: 768
- **Features**: 8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
- **Classes**: 2 (No Diabetes: 0, Diabetes: 1)
- **Classes**: We have also used a synthetic dataset 

