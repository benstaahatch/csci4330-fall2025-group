# ------------------------------------------------------------
# Hybrid MPI + OpenMP Random Forest — Makefile (macOS)
# ------------------------------------------------------------

# Use system clang++ with Homebrew libomp
CXX = clang++
OMP_LIB = /opt/homebrew/opt/libomp/lib

# Use mpicxx directly
MPI_CXX = mpicxx

CXXFLAGS = -O3 -std=c++17 -march=native -Xpreprocessor -fopenmp -Iinclude -I/opt/homebrew/opt/libomp/include
LDFLAGS  = -L$(OMP_LIB) -lomp

# Supported executables
MPI_FOREST_TARGET = mpi_forest
HYBRID_RF_TARGET = hybrid_rf
BACKUP_RF_SERIAL_TARGET = backup_rf_serial_compare
BACKUP_RF_OPENMP_TARGET = backup_rf_openmp_compare
BACKUP_RF_MPI_TARGET = backup_rf_mpi_compare

# Shared source files for the hybrid build
COMMON_SRC = src/bins.cpp \
             src/histogram.cpp \
             src/tree_builder.cpp \
             src/forest.cpp

# Canonical hybrid entry point
MPI_FOREST_SRC = src/main.cpp \
                 $(COMMON_SRC) \
                 src/mpi_forest.cpp
BACKUP_RF_SERIAL_SRC = src/backup_rf_serial_compare.cpp
BACKUP_RF_OPENMP_SRC = src/backup_rf_openmp_compare.cpp
BACKUP_RF_MPI_SRC = src/backup_rf_mpi_compare.cpp

# ------------------------------------------------------------
# Build supported executables
# ------------------------------------------------------------
all: $(MPI_FOREST_TARGET)

# MPI + OpenMP hybrid build
$(MPI_FOREST_TARGET): $(MPI_FOREST_SRC)
	$(MPI_CXX) $(CXXFLAGS) $(MPI_FOREST_SRC) $(LDFLAGS) -o $(MPI_FOREST_TARGET)
	@echo "Built: ./$(MPI_FOREST_TARGET) (MPI + OpenMP)"

# README-compatible alias
$(HYBRID_RF_TARGET): $(MPI_FOREST_TARGET)
	cp $(MPI_FOREST_TARGET) $(HYBRID_RF_TARGET)
	@echo "Built: ./$(HYBRID_RF_TARGET) (alias of MPI + OpenMP build)"

$(BACKUP_RF_SERIAL_TARGET): $(BACKUP_RF_SERIAL_SRC)
	$(CXX) $(CXXFLAGS) $(BACKUP_RF_SERIAL_SRC) $(LDFLAGS) -o $(BACKUP_RF_SERIAL_TARGET)
	@echo "Built: ./$(BACKUP_RF_SERIAL_TARGET)"

$(BACKUP_RF_OPENMP_TARGET): $(BACKUP_RF_OPENMP_SRC)
	$(CXX) $(CXXFLAGS) $(BACKUP_RF_OPENMP_SRC) $(LDFLAGS) -o $(BACKUP_RF_OPENMP_TARGET)
	@echo "Built: ./$(BACKUP_RF_OPENMP_TARGET)"

$(BACKUP_RF_MPI_TARGET): $(BACKUP_RF_MPI_SRC)
	$(MPI_CXX) $(CXXFLAGS) $(BACKUP_RF_MPI_SRC) $(LDFLAGS) -o $(BACKUP_RF_MPI_TARGET)
	@echo "Built: ./$(BACKUP_RF_MPI_TARGET)"

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
clean:
	rm -f $(MPI_FOREST_TARGET) $(HYBRID_RF_TARGET) $(BACKUP_RF_SERIAL_TARGET) $(BACKUP_RF_OPENMP_TARGET) $(BACKUP_RF_MPI_TARGET)
	@echo "Clean complete."

# ------------------------------------------------------------
# Run targets
# ------------------------------------------------------------
NP ?= 2
SEEDS ?= 7 21 42 84 123
RUN_ARGS ?=

run-mpi: $(MPI_FOREST_TARGET)
	mpirun -np $(NP) ./$(MPI_FOREST_TARGET)

run-hybrid: $(HYBRID_RF_TARGET)
	mpirun -np $(NP) ./$(HYBRID_RF_TARGET)

run-backup-serial: $(BACKUP_RF_SERIAL_TARGET)
	./$(BACKUP_RF_SERIAL_TARGET)

run-backup-openmp: $(BACKUP_RF_OPENMP_TARGET)
	./$(BACKUP_RF_OPENMP_TARGET) --threads 4

run-backup-mpi: $(BACKUP_RF_MPI_TARGET)
	mpirun -np $(NP) ./$(BACKUP_RF_MPI_TARGET)

compare-backup-baselines: $(BACKUP_RF_SERIAL_TARGET) $(BACKUP_RF_OPENMP_TARGET) $(MPI_FOREST_TARGET)
	./$(BACKUP_RF_SERIAL_TARGET) --trees 100 --max-depth 10 --min-samples-split 2
	./$(BACKUP_RF_OPENMP_TARGET) --threads 4 --trees 100 --max-depth 10 --min-samples-split 2
	mpirun -np 2 ./$(MPI_FOREST_TARGET) --trees 100 --max-depth 10 --min-samples-split 2
	python3 Results/compare_backup_baselines.py

sweep-seeds: $(MPI_FOREST_TARGET)
	@for seed in $(SEEDS); do \
			echo "Running split seed $$seed"; \
			mpirun -np $(NP) ./$(MPI_FOREST_TARGET) --split-seed $$seed $(RUN_ARGS); \
		done

summarize-runs:
	python3 Results/summarize_runs.py

# Visualization - Compare Models
visualize:
	@echo "Running models and generating comparison visualizations..."
	@python3 -c "import matplotlib, numpy" 2>/dev/null && \
		python3 Results/compare_models.py || \
		echo "Error: Please install matplotlib and numpy: pip3 install --break-system-packages matplotlib numpy"

# Default run
run: run-mpi

.PHONY: all clean run run-mpi run-hybrid run-backup-serial run-backup-openmp run-backup-mpi compare-backup-baselines sweep-seeds summarize-runs visualize
