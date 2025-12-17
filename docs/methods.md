# Methods

## How It Works

1. **Data Distribution**: Rank 0 loads the dataset and broadcasts it to all MPI ranks
2. **Tree Distribution**: Trees are evenly split across MPI ranks (e.g., 500 trees with 2 ranks = 250 trees per rank)
3. **Parallel Training**: Each rank trains its subset of trees using OpenMP for parallel tree building
4. **Vote Aggregation**: Predictions from all ranks are combined using MPI_Allreduce to get final predictions


