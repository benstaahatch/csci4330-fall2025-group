#ifndef MPI_FOREST_H
#define MPI_FOREST_H

#include <vector>
#include <cstdint>
#include <mpi.h>
#include "forest.h"
using namespace std;

// ------------------------------------------------------------
// MPI Random Forest wrapper.
// Each rank trains a subset of trees using OpenMP locally.
// MPI handles combining predictions across ranks.
// ------------------------------------------------------------
class MPIRandomForest {
private:
    const vector<vector<uint8_t>>& X_bins;
    const vector<int>& y;
    ForestConfig cfg;

    int rank;
    int size;

    int treesPerRank;
    int startTree;
    int endTree;

    RandomForest* localForest;

public:
    MPIRandomForest(
        const vector<vector<uint8_t>>& X_bins,
        const vector<int>& labels,
        const ForestConfig& cfg
    );

    // Train forest in parallel across MPI ranks
    void fit();

    // Predict an arbitrary evaluation dataset via MPI vote merging
    vector<int> predict(const vector<vector<uint8_t>>& X_eval_bins);

};

#endif
