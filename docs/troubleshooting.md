# Troubleshooting

### Build Issues

If you encounter OpenMP errors on macOS:
```bash
# Update Makefile paths for your system
# Check OpenMP library location:
find /opt/homebrew -name "libomp*" 2>/dev/null
```

### Runtime Issues

If MPI fails to run:
```bash
# Check MPI installation
which mpirun
mpirun --version

# Try with explicit path
/opt/homebrew/bin/mpirun -np 2 ./mpi_forest
```




