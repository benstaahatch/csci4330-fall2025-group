#include <iostream>
#include <map>
#include <mpi.h>
#include <random>
#include <set>
#include <vector>
#include "backup_compare_utils.h"

using namespace std;

struct TreeNode {
    bool isLeaf;
    int predictedClass;
    int featureIndex;
    double threshold;
    TreeNode* left;
    TreeNode* right;

    TreeNode()
        : isLeaf(false), predictedClass(0), featureIndex(-1),
          threshold(0.0), left(nullptr), right(nullptr) {}

    ~TreeNode()
    {
        delete left;
        delete right;
    }
};

class DecisionTree {
private:
    int maxDepth;
    int minSamplesSplit;
    TreeNode* root;
    mt19937 rng;

    double giniImpurity(const vector<int>& labels)
    {
        if (labels.empty()) return 0.0;

        map<int, int> counts;
        for (int label : labels) {
            counts[label]++;
        }

        double impurity = 1.0;
        for (const auto& pair : counts) {
            double prob = static_cast<double>(pair.second) / labels.size();
            impurity -= prob * prob;
        }
        return impurity;
    }

    int mostCommonClass(const vector<int>& labels)
    {
        map<int, int> counts;
        for (int label : labels) {
            counts[label]++;
        }

        int maxCount = 0;
        int commonClass = 0;
        for (const auto& pair : counts) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                commonClass = pair.first;
            }
        }
        return commonClass;
    }

    void splitData(
        const vector<vector<double>>& X,
        const vector<int>& y,
        int featureIdx,
        double threshold,
        vector<vector<double>>& X_left,
        vector<int>& y_left,
        vector<vector<double>>& X_right,
        vector<int>& y_right
    )
    {
        for (size_t i = 0; i < X.size(); ++i) {
            if (X[i][featureIdx] <= threshold) {
                X_left.push_back(X[i]);
                y_left.push_back(y[i]);
            } else {
                X_right.push_back(X[i]);
                y_right.push_back(y[i]);
            }
        }
    }

    bool findBestSplit(
        const vector<vector<double>>& X,
        const vector<int>& y,
        const vector<int>& featureIndices,
        int& bestFeature,
        double& bestThreshold
    )
    {
        double bestGain = -1.0;
        double currentImpurity = giniImpurity(y);
        int sampleCount = static_cast<int>(X.size());

        for (int featureIdx : featureIndices) {
            set<double> uniqueVals;
            for (const auto& sample : X) {
                uniqueVals.insert(sample[featureIdx]);
            }

            for (double threshold : uniqueVals) {
                vector<vector<double>> X_left, X_right;
                vector<int> y_left, y_right;
                splitData(X, y, featureIdx, threshold, X_left, y_left, X_right, y_right);

                if (y_left.empty() || y_right.empty()) continue;

                double n_left = static_cast<double>(y_left.size());
                double n_right = static_cast<double>(y_right.size());
                double weightedImpurity =
                    (n_left / sampleCount) * giniImpurity(y_left) +
                    (n_right / sampleCount) * giniImpurity(y_right);
                double gain = currentImpurity - weightedImpurity;

                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = featureIdx;
                    bestThreshold = threshold;
                }
            }
        }

        return bestGain > 0.0;
    }

    TreeNode* buildTree(
        const vector<vector<double>>& X,
        const vector<int>& y,
        int depth,
        const vector<int>& featureIndices
    )
    {
        TreeNode* node = new TreeNode();

        if (depth >= maxDepth ||
            y.size() < static_cast<size_t>(minSamplesSplit) ||
            giniImpurity(y) == 0.0) {
            node->isLeaf = true;
            node->predictedClass = mostCommonClass(y);
            return node;
        }

        int bestFeature = -1;
        double bestThreshold = 0.0;
        if (!findBestSplit(X, y, featureIndices, bestFeature, bestThreshold)) {
            node->isLeaf = true;
            node->predictedClass = mostCommonClass(y);
            return node;
        }

        vector<vector<double>> X_left, X_right;
        vector<int> y_left, y_right;
        splitData(X, y, bestFeature, bestThreshold, X_left, y_left, X_right, y_right);

        node->featureIndex = bestFeature;
        node->threshold = bestThreshold;
        node->left = buildTree(X_left, y_left, depth + 1, featureIndices);
        node->right = buildTree(X_right, y_right, depth + 1, featureIndices);
        return node;
    }

    int predictSample(const vector<double>& x, TreeNode* node) const
    {
        if (node->isLeaf) {
            return node->predictedClass;
        }
        if (x[node->featureIndex] <= node->threshold) {
            return predictSample(x, node->left);
        }
        return predictSample(x, node->right);
    }

public:
    DecisionTree(int maxDepth, int minSamplesSplit, int seed)
        : maxDepth(maxDepth), minSamplesSplit(minSamplesSplit), root(nullptr), rng(seed) {}

    ~DecisionTree()
    {
        delete root;
    }

    void fit(const vector<vector<double>>& X, const vector<int>& y, const vector<int>& featureIndices)
    {
        root = buildTree(X, y, 0, featureIndices);
    }

    int predictOne(const vector<double>& x) const
    {
        return predictSample(x, root);
    }
};

class LocalRandomForest {
private:
    int nEstimators;
    int maxDepth;
    int minSamplesSplit;
    int maxFeatures;
    bool bootstrap;
    vector<DecisionTree*> trees;

    void getBootstrapSample(
        const vector<vector<double>>& X,
        const vector<int>& y,
        vector<vector<double>>& X_sample,
        vector<int>& y_sample,
        int seed
    )
    {
        mt19937 local_rng(seed);
        uniform_int_distribution<int> dist(0, static_cast<int>(X.size()) - 1);
        for (size_t i = 0; i < X.size(); ++i) {
            int idx = dist(local_rng);
            X_sample.push_back(X[idx]);
            y_sample.push_back(y[idx]);
        }
    }

    vector<int> getRandomFeatures(int nFeatures, int seed)
    {
        vector<int> allFeatures(nFeatures);
        iota(allFeatures.begin(), allFeatures.end(), 0);
        mt19937 local_rng(seed);
        shuffle(allFeatures.begin(), allFeatures.end(), local_rng);

        int n_select = min(maxFeatures, nFeatures);
        return vector<int>(allFeatures.begin(), allFeatures.begin() + n_select);
    }

public:
    LocalRandomForest(
        int nEstimators,
        int maxDepth,
        int minSamplesSplit,
        int maxFeatures,
        bool bootstrap
    )
        : nEstimators(nEstimators),
          maxDepth(maxDepth),
          minSamplesSplit(minSamplesSplit),
          maxFeatures(maxFeatures),
          bootstrap(bootstrap)
    {}

    ~LocalRandomForest()
    {
        for (auto* tree : trees) {
            delete tree;
        }
    }

    void fit(const vector<vector<double>>& X, const vector<int>& y)
    {
        int nFeatures = static_cast<int>(X[0].size());
        if (maxFeatures == -1) {
            maxFeatures = static_cast<int>(sqrt(nFeatures));
        }

        cout << "Local trees on this rank: " << nEstimators << "\n";
        trees.reserve(nEstimators);
        for (int i = 0; i < nEstimators; ++i) {
            vector<vector<double>> X_sample;
            vector<int> y_sample;
            if (bootstrap) {
                getBootstrapSample(X, y, X_sample, y_sample, 42 + i);
            } else {
                X_sample = X;
                y_sample = y;
            }

            vector<int> featureIndices = getRandomFeatures(nFeatures, 42 + i);
            auto* tree = new DecisionTree(maxDepth, minSamplesSplit, 42 + i);
            tree->fit(X_sample, y_sample, featureIndices);
            trees.push_back(tree);
        }
    }

    vector<int> predictVoteCounts(const vector<vector<double>>& X_eval) const
    {
        vector<int> positiveVotes(X_eval.size(), 0);
        for (const auto* tree : trees) {
            for (size_t i = 0; i < X_eval.size(); ++i) {
                positiveVotes[i] += tree->predictOne(X_eval[i]);
            }
        }
        return positiveVotes;
    }
};

int main(int argc, char* argv[])
{
    CompareExperimentConfig experiment;
    if (!parseCompareArgs(argc, argv, experiment)) {
        return 1;
    }

    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<vector<double>> X_raw;
    vector<int> y;

    if (rank == 0) {
        cout << "============================================================\n";
        cout << "Backup MPI Random Forest With Current Validation Protocol\n";
        cout << "============================================================\n";
        readCSV("diabetes.csv", X_raw, y);
        cout << "Dataset loaded: " << X_raw.size() << " samples, "
             << X_raw[0].size() << " features\n";
    }

    int n_samples = 0;
    int n_features = 0;
    if (rank == 0) {
        n_samples = static_cast<int>(X_raw.size());
        n_features = static_cast<int>(X_raw[0].size());
    }

    MPI_Bcast(&n_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_features, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        y.resize(n_samples);
    }
    MPI_Bcast(y.data(), n_samples, MPI_INT, 0, MPI_COMM_WORLD);

    vector<vector<double>> X_local;
    if (rank == 0) {
        X_local = X_raw;
    } else {
        X_local.assign(n_samples, vector<double>(n_features));
    }

    for (int j = 0; j < n_features; ++j) {
        vector<double> col(n_samples);
        if (rank == 0) {
            for (int i = 0; i < n_samples; ++i) {
                col[i] = X_raw[i][j];
            }
        }
        MPI_Bcast(col.data(), n_samples, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int i = 0; i < n_samples; ++i) {
            X_local[i][j] = col[i];
        }
    }

    vector<int> train_idx;
    vector<int> val_idx;
    stratifiedTrainValidationSplit(y, experiment.trainFraction, experiment.splitSeed, train_idx, val_idx);

    vector<vector<double>> X_train;
    vector<vector<double>> X_val;
    vector<int> y_train;
    vector<int> y_val;
    extractSubset(X_local, y, train_idx, X_train, y_train);
    extractSubset(X_local, y, val_idx, X_val, y_val);

    auto medians = computeInvalidZeroMedians(X_train, n_features);
    imputeInvalidZeros(X_train, medians);
    imputeInvalidZeros(X_val, medians);

    MinMaxStats scaleStats = computeMinMaxStats(X_train);
    applyMinMaxScale(X_train, scaleStats);
    applyMinMaxScale(X_val, scaleStats);

    if (rank == 0) {
        cout << "Train/validation split: " << X_train.size() << " train, "
             << X_val.size() << " validation\n";
    }

    int baseTrees = experiment.nTrees / size;
    int remainder = experiment.nTrees % size;
    int localTrees = baseTrees + (rank < remainder ? 1 : 0);

    auto start = chrono::high_resolution_clock::now();

    LocalRandomForest forest(
        localTrees,
        experiment.maxDepth,
        experiment.minSamplesSplit,
        experiment.maxFeatures,
        true
    );
    forest.fit(X_train, y_train);

    vector<int> localPositiveVotes = forest.predictVoteCounts(X_val);
    vector<int> globalPositiveVotes(X_val.size(), 0);
    MPI_Allreduce(
        localPositiveVotes.data(),
        globalPositiveVotes.data(),
        static_cast<int>(X_val.size()),
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    vector<int> predictions(X_val.size(), 0);
    for (size_t i = 0; i < X_val.size(); ++i) {
        predictions[i] = (globalPositiveVotes[i] * 2 >= experiment.nTrees) ? 1 : 0;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    if (rank == 0) {
        double endMemory = getMemoryUsageMB();
        CompareMetrics metrics = computeMetrics(y_val, predictions);
        cout << "\n---------------------------------------------------\n";
        cout << "T_mpi (seconds): " << elapsed.count() << "\n";
        cout << "Memory Usage (MB): " << endMemory << "\n";
        cout << "MPI Ranks: " << size << "\n";
        cout << "---------------------------------------------------\n";
        printMetrics(metrics);
        cout << "---------------------------------------------------\n";

        string run_dir = makeRunDirectory(experiment, "backup_mpi_rf", size, 1);
        writeMetricsJson(
            run_dir,
            "Random Forest (Backup MPI)",
            metrics,
            static_cast<int>(X_train.size()),
            static_cast<int>(X_val.size()),
            size,
            1,
            elapsed.count(),
            experiment
        );
        cout << "Saved metrics to: " << run_dir << "/metrics.json\n";
    }

    MPI_Finalize();
    return 0;
}
