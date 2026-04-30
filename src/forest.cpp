#include "forest.h"
#include <iostream>
#include <omp.h>
#include <algorithm>

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------
RandomForest::RandomForest(
    const vector<vector<uint8_t>>& X_bins_,
    const vector<int>& labels_,
    const ForestConfig& cfg_
)
: X_bins(X_bins_), y(labels_), cfg(cfg_), rng(cfg.seed)
{
    trees.resize(cfg.nTrees, nullptr);
}

// ------------------------------------------------------------
// Bootstrap sampling (same idea as your OpenMP version)
// ------------------------------------------------------------
vector<int> RandomForest::bootstrapSample(int n_samples, int seed)
{
    vector<int> sample(n_samples);
    mt19937 local_rng(seed);
    uniform_int_distribution<int> dist(0, n_samples - 1);

    for (int i = 0; i < n_samples; ++i) {
        sample[i] = dist(local_rng);
    }
    return sample;
}

// ------------------------------------------------------------
// Train forest using OpenMP
// ------------------------------------------------------------
void RandomForest::fit()
{
    int n_samples = y.size();
    int n_features = X_bins.size();

    // Tree config for builder
    TreeConfig tcfg;
    tcfg.maxDepth        = cfg.maxDepth;
    tcfg.minSamplesSplit = cfg.minSamplesSplit;
    tcfg.maxFeatures     = cfg.maxFeatures;
    tcfg.numClasses      = cfg.numClasses;

    // Parallel training of trees
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < cfg.nTrees; ++t) {

        int seed_t = cfg.seed + t;

        // Bootstrap rows for this tree
        vector<int> bootIdx = bootstrapSample(n_samples, seed_t);

        // Create tree builder
        tcfg.seed = seed_t;
        DecisionTreeBuilder builder(X_bins, y, tcfg);

        // Train tree
        TreeNode* root = builder.buildTree(bootIdx);

        // Store tree pointer
        trees[t] = root;

        // Optional progress print (only from one thread)
        #pragma omp critical
        {
            if ((t + 1) % 10 == 0)
                cout << "Trained " << (t + 1) << "/" << cfg.nTrees << " trees\n";
        }
    }
}

// ------------------------------------------------------------
// Predict one sample using all trees (majority vote).
// sampleCols[j] is a pointer to X_bins[j][row]
// ------------------------------------------------------------
int RandomForest::predictSingle(const vector<uint8_t*>& sampleCols) const
{
    vector<int> votes(cfg.numClasses, 0);

    for (TreeNode* root : trees) {
        TreeNode* node = root;

        // Walk down the tree
        while (!node->isLeaf) {
            uint8_t bin = *sampleCols[node->featureIndex];
            if (bin <= node->binThreshold)
                node = node->left;
            else
                node = node->right;
        }

        votes[node->predictedClass]++;
    }

    // Majority class
    int best = 0;
    for (int c = 1; c < cfg.numClasses; ++c) {
        if (votes[c] > votes[best])
            best = c;
    }
    return best;
}

// ------------------------------------------------------------
// Predict over all rows in an arbitrary evaluation dataset.
// Uses OpenMP to parallelize over samples.
// ------------------------------------------------------------
vector<int> RandomForest::predictBatch(const vector<vector<uint8_t>>& X_eval_bins) const
{
    int n_samples = X_eval_bins.empty() ? 0 : X_eval_bins[0].size();
    vector<int> voteCounts = predictVoteCounts(X_eval_bins);
    vector<int> preds(n_samples, 0);

    for (int i = 0; i < n_samples; ++i) {
        int bestClass = 0;
        int bestVotes = voteCounts[i * cfg.numClasses];
        for (int c = 1; c < cfg.numClasses; ++c) {
            int classVotes = voteCounts[i * cfg.numClasses + c];
            if (classVotes > bestVotes) {
                bestVotes = classVotes;
                bestClass = c;
            }
        }
        preds[i] = bestClass;
    }

    return preds;
}

// ------------------------------------------------------------
// Predict raw vote counts over all rows in an arbitrary evaluation
// dataset. Each tree contributes one vote per sample.
// ------------------------------------------------------------
vector<int> RandomForest::predictVoteCounts(const vector<vector<uint8_t>>& X_eval_bins) const
{
    int n_samples = X_eval_bins.empty() ? 0 : X_eval_bins[0].size();
    int n_features = X_eval_bins.size();

    vector<int> voteCounts(n_samples * cfg.numClasses, 0);

    #pragma omp parallel for
    for (int i = 0; i < n_samples; ++i) {
        vector<uint8_t*> cols(n_features);
        for (int j = 0; j < n_features; ++j)
            cols[j] = (uint8_t*)&X_eval_bins[j][i];

        vector<int> localVotes(cfg.numClasses, 0);
        for (TreeNode* root : trees) {
            TreeNode* node = root;
            while (!node->isLeaf) {
                uint8_t bin = *cols[node->featureIndex];
                if (bin <= node->binThreshold)
                    node = node->left;
                else
                    node = node->right;
            }
            localVotes[node->predictedClass]++;
        }

        for (int c = 0; c < cfg.numClasses; ++c)
            voteCounts[i * cfg.numClasses + c] = localVotes[c];
    }

    return voteCounts;
}
