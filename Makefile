# ------------------------------------------------------------
# Hybrid MPI + OpenMP Random Forest — Makefile
# Multi-Architecture Support: Apple Silicon & Intel macOS
# ------------------------------------------------------------

# Detect architecture (default target)
ARCH := $(shell uname -m)

# Apple Silicon paths
SILICON_LLVM_CLANG = /Users/jhatcher/.swiftly/bin/clang++
SILICON_LLVM_OMP   = /opt/homebrew/opt/libomp

# Intel Mac paths
INTEL_LLVM_CLANG = /Users/jhatcher/.swiftly/bin/clang++
INTEL_LLVM_OMP   = /usr/local/opt/libomp

# Set default based on detected architecture
ifeq ($(ARCH),arm64)
    LLVM_CLANG = $(SILICON_LLVM_CLANG)
    LLVM_OMP   = $(SILICON_LLVM_OMP)
    ARCH_TYPE  = Apple Silicon (arm64)
else
    LLVM_CLANG = $(INTEL_LLVM_CLANG)
    LLVM_OMP   = $(INTEL_LLVM_OMP)
    ARCH_TYPE  = Intel (x86_64)
endif

# Force mpicxx to use LLVM
MPI_CXX = OMPI_CXX=$(LLVM_CLANG) mpicxx

CXXFLAGS = -O3 -std=c++17 -march=native -fopenmp -Iinclude -I$(LLVM_OMP)/include
LDFLAGS  = -L$(LLVM_OMP)/lib -lomp

TARGET = hybrid_rf

SRC = src/main.cpp \
      src/bins.cpp \
      src/histogram.cpp \
      src/tree_builder.cpp \
      src/forest.cpp \
      src/mpi_forest.cpp

# ------------------------------------------------------------
# Build Targets
# ------------------------------------------------------------

# Default build (auto-detect architecture)
all: $(TARGET)

$(TARGET): $(SRC)
	@echo "Building for $(ARCH_TYPE)..."
	$(MPI_CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)
	@echo "------------------------------------------------------------"
	@echo "Build complete: ./$(TARGET)"
	@echo "Architecture: $(ARCH_TYPE)"
	@echo "Use 'make run' to execute with MPI"
	@echo "------------------------------------------------------------"

# Explicit Apple Silicon build
silicon:
	@echo "Building for Apple Silicon (arm64)..."
	$(MAKE) all \
		LLVM_CLANG=$(SILICON_LLVM_CLANG) \
		LLVM_OMP=$(SILICON_LLVM_OMP) \
		ARCH_TYPE="Apple Silicon (arm64)"

# Explicit Intel Mac build
intel:
	@echo "Building for Intel Mac (x86_64)..."
	$(MAKE) all \
		LLVM_CLANG=$(INTEL_LLVM_CLANG) \
		LLVM_OMP=$(INTEL_LLVM_OMP) \
		ARCH_TYPE="Intel Mac (x86_64)"

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

.PHONY: all clean run silicon intel
