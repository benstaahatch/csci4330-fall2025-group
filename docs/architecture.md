## Architecture

- **MPI Layer**: Handles distributed computing across multiple processes
- **OpenMP Layer**: Parallelizes tree building within each MPI rank
- **Histogram-based Splits**: Fast split finding using binned data representation
- **Bootstrap Sampling**: Each tree uses a random bootstrap sample of the training data


