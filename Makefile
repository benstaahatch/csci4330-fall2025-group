# ------------------------------------------------------------
# Hybrid MPI + OpenMP Random Forest — Makefile (Intel macOS)
# ------------------------------------------------------------

LLVM_CLANG = /usr/local/opt/llvm/bin/clang++
LLVM_OMP   = /usr/local/opt/llvm/lib

# Force mpicxx to use Homebrew LLVM
MPI_CXX = OMPI_CXX=$(LLVM_CLANG) mpicxx

CXXFLAGS = -O3 -std=c++17 -march=native -fopenmp -Iinclude
LDFLAGS  = -L$(LLVM_OMP) -lomp

TARGET = hybrid_rf

SRC = src/main.cpp \
      src/bins.cpp \
      src/histogram.cpp \
      src/tree_builder.cpp \
      src/forest.cpp \
      src/mpi_forest.cpp

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------
all: $(TARGET)

$(TARGET): $(SRC)
	$(MPI_CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)
	@echo "------------------------------------------------------------"
	@echo "Build complete: ./$(TARGET)"
	@echo "Use 'make run' to execute with MPI"
	@echo "------------------------------------------------------------"

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
clean:
	rm -f $(TARGET)
	@echo "Clean complete."

# ------------------------------------------------------------
# Run (default 4 MPI ranks)
# ------------------------------------------------------------
NP ?= 4

run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET)

.PHONY: all clean run
