#ifndef FOREST_H
#define FOREST_H

#include <vector>
#include <random>
#include <cstdint>
#include "tree_builder.h"
using namespace std;

// ------------------------------------------------------------
// Random Forest configuration
// ------------------------------------------------------------
struct ForestConfig {
    int nTrees;
    int maxDepth;
    int minSamplesSplit;
    int maxFeatures;
    int numClasses;
    int seed;
};

// ------------------------------------------------------------
// Random Forest (local version, used per MPI rank).
// Trains multiple trees using OpenMP + TreeBuilder.
// ------------------------------------------------------------
class RandomForest {
private:
    const vector<vector<uint8_t>>& X_bins;
    const vector<int>& y;
    ForestConfig cfg;
    vector<TreeNode*> trees;
    mt19937 rng;

    // Bootstrap sampling: returns vector of row indices
    vector<int> bootstrapSample(int n_samples, int seed);

public:
    RandomForest(
        const vector<vector<uint8_t>>& X_bins,
        const vector<int>& labels,
        const ForestConfig& cfg
    );

    // Train forest using OpenMP
    void fit();

    // Predict a single row
    int predictSingle(const vector<uint8_t*>& sampleCols) const;

    // Predict a batch on an arbitrary binned dataset
    vector<int> predictBatch(const vector<vector<uint8_t>>& X_eval_bins) const;

    // Predict vote counts for each sample/class across all local trees
    vector<int> predictVoteCounts(const vector<vector<uint8_t>>& X_eval_bins) const;

    // Access trained trees
    const vector<TreeNode*>& getTrees() const {
        return trees;
    }
};

#endif
