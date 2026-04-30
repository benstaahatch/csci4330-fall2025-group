#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <ctime>
#include <cmath>
#include <filesystem>
#include <mpi.h>
#include <sys/resource.h>   // For memory usage
#include <omp.h>            // For OpenMP thread count
#include "bins.h"
#include "mpi_forest.h"
using namespace std;

struct Metrics {
    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double overhead = 0.0;
};

struct ExperimentConfig {
    double trainFraction = 0.8;
    int splitSeed = 42;
    int nTrees = 500;
    int maxDepth = 20;
    int minSamplesSplit = 5;
    int maxFeatures = -1;  // -1 => use sqrt(n_features)
};

const vector<int> INVALID_ZERO_COLUMNS = {1, 2, 3, 4, 5};

void printUsage(const char* programName)
{
    cout << "Usage: " << programName
         << " [--train-fraction FLOAT] [--split-seed INT] [--trees INT]"
         << " [--max-depth INT] [--min-samples-split INT] [--max-features INT]\n";
}

bool parseArgs(int argc, char* argv[], ExperimentConfig& config)
{
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        auto requireValue = [&](const string& name) -> bool {
            if (i + 1 >= argc) {
                cerr << "Missing value for " << name << "\n";
                return false;
            }
            return true;
        };

        if (arg == "--train-fraction") {
            if (!requireValue(arg)) return false;
            config.trainFraction = stod(argv[++i]);
        } else if (arg == "--split-seed") {
            if (!requireValue(arg)) return false;
            config.splitSeed = stoi(argv[++i]);
        } else if (arg == "--trees") {
            if (!requireValue(arg)) return false;
            config.nTrees = stoi(argv[++i]);
        } else if (arg == "--max-depth") {
            if (!requireValue(arg)) return false;
            config.maxDepth = stoi(argv[++i]);
        } else if (arg == "--min-samples-split") {
            if (!requireValue(arg)) return false;
            config.minSamplesSplit = stoi(argv[++i]);
        } else if (arg == "--max-features") {
            if (!requireValue(arg)) return false;
            config.maxFeatures = stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        } else {
            cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return false;
        }
    }

    if (config.trainFraction <= 0.0 || config.trainFraction >= 1.0) {
        cerr << "--train-fraction must be between 0 and 1.\n";
        return false;
    }
    if (config.nTrees <= 0) {
        cerr << "--trees must be positive.\n";
        return false;
    }
    if (config.maxDepth <= 0) {
        cerr << "--max-depth must be positive.\n";
        return false;
    }
    if (config.minSamplesSplit <= 1) {
        cerr << "--min-samples-split must be greater than 1.\n";
        return false;
    }
    if (config.maxFeatures == 0 || config.maxFeatures < -1) {
        cerr << "--max-features must be -1 (auto) or a positive integer.\n";
        return false;
    }

    return true;
}

// ------------------------------------------------------------
// Load CSV file as floats. Last column is class label.
// ------------------------------------------------------------
void readCSV(const string& filename, vector<vector<double>>& X, vector<int>& y)
{
    ifstream file(filename);
    string line;
    bool skipHeader = true;

    while (getline(file, line)) {
        if (skipHeader) { skipHeader = false; continue; }

        stringstream ss(line);
        string value;
        vector<double> row;

        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }

        int cls = (int)row.back();
        row.pop_back();

        X.push_back(row);
        y.push_back(cls);
    }
    file.close();
}

// ------------------------------------------------------------
// Deterministic stratified train/validation split of row indices
// ------------------------------------------------------------
void stratifiedTrainValidationSplit(
    const vector<int>& y,
    double train_fraction,
    int seed,
    vector<int>& train_idx,
    vector<int>& val_idx
)
{
    mt19937 rng(seed);
    vector<int> class0;
    vector<int> class1;

    for (int i = 0; i < y.size(); ++i) {
        if (y[i] == 0) class0.push_back(i);
        else class1.push_back(i);
    }

    shuffle(class0.begin(), class0.end(), rng);
    shuffle(class1.begin(), class1.end(), rng);

    auto splitClass = [&](const vector<int>& class_indices) {
        int split_point = static_cast<int>(class_indices.size() * train_fraction);
        train_idx.insert(train_idx.end(), class_indices.begin(), class_indices.begin() + split_point);
        val_idx.insert(val_idx.end(), class_indices.begin() + split_point, class_indices.end());
    };

    train_idx.clear();
    val_idx.clear();
    splitClass(class0);
    splitClass(class1);

    shuffle(train_idx.begin(), train_idx.end(), rng);
    shuffle(val_idx.begin(), val_idx.end(), rng);
}

// ------------------------------------------------------------
// Extract subset rows by index
// ------------------------------------------------------------
void extractSubset(
    const vector<vector<double>>& X,
    const vector<int>& y,
    const vector<int>& indices,
    vector<vector<double>>& X_subset,
    vector<int>& y_subset
)
{
    X_subset.clear();
    y_subset.clear();
    X_subset.reserve(indices.size());
    y_subset.reserve(indices.size());

    for (int idx : indices) {
        X_subset.push_back(X[idx]);
        y_subset.push_back(y[idx]);
    }
}

// ------------------------------------------------------------
// Compute per-feature median from non-zero training values only
// for columns where zero is treated as invalid/missing.
// ------------------------------------------------------------
vector<double> computeInvalidZeroMedians(
    const vector<vector<double>>& X_train,
    int n_features
)
{
    vector<double> medians(n_features, 0.0);

    for (int col : INVALID_ZERO_COLUMNS) {
        vector<double> values;
        values.reserve(X_train.size());

        for (const auto& row : X_train) {
            if (row[col] != 0.0)
                values.push_back(row[col]);
        }

        if (values.empty())
            continue;

        size_t mid = values.size() / 2;
        nth_element(values.begin(), values.begin() + mid, values.end());
        double median = values[mid];

        if (values.size() % 2 == 0) {
            nth_element(values.begin(), values.begin() + mid - 1, values.end());
            median = 0.5 * (median + values[mid - 1]);
        }

        medians[col] = median;
    }

    return medians;
}

// ------------------------------------------------------------
// Replace invalid zero entries with training-derived medians.
// ------------------------------------------------------------
void imputeInvalidZeros(
    vector<vector<double>>& X,
    const vector<double>& medians
)
{
    for (auto& row : X) {
        for (int col : INVALID_ZERO_COLUMNS) {
            if (row[col] == 0.0 && medians[col] > 0.0)
                row[col] = medians[col];
        }
    }
}

// ------------------------------------------------------------
// Compute confusion matrix and classification metrics
// ------------------------------------------------------------
Metrics computeMetrics(const vector<int>& y, const vector<int>& pred)
{
    Metrics metrics;

    for (int i = 0; i < y.size(); ++i) {
        if (pred[i] == 1 && y[i] == 1) metrics.tp++;
        else if (pred[i] == 0 && y[i] == 0) metrics.tn++;
        else if (pred[i] == 1 && y[i] == 0) metrics.fp++;
        else metrics.fn++;
    }

    int total = metrics.tp + metrics.tn + metrics.fp + metrics.fn;
    metrics.accuracy = (total == 0) ? 0.0 : static_cast<double>(metrics.tp + metrics.tn) / total;
    metrics.precision = (metrics.tp + metrics.fp == 0) ? 0.0 : static_cast<double>(metrics.tp) / (metrics.tp + metrics.fp);
    metrics.recall = (metrics.tp + metrics.fn == 0) ? 0.0 : static_cast<double>(metrics.tp) / (metrics.tp + metrics.fn);
    metrics.f1 = (metrics.precision + metrics.recall == 0.0) ? 0.0 :
                 2.0 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);
    metrics.overhead = 1.0 - metrics.accuracy;

    return metrics;
}

// ------------------------------------------------------------
// Print confusion matrix & accuracy
// ------------------------------------------------------------
void evaluateMetrics(const Metrics& metrics)
{
    cout << "\n---- Metrics ----\n";
    cout << "TP: " << metrics.tp << "  TN: " << metrics.tn << "\n";
    cout << "FP: " << metrics.fp << "  FN: " << metrics.fn << "\n";
    cout << "Accuracy: " << metrics.accuracy * 100 << "%\n";
}

// ------------------------------------------------------------
// Full metrics including Overhead instead of Specificity
// ------------------------------------------------------------
void fullMetrics(const Metrics& metrics)
{
    cout << "\n===== FULL MODEL METRICS =====\n";
    cout << "Accuracy:    " << metrics.accuracy * 100 << "%\n";
    cout << "Precision:   " << metrics.precision * 100 << "%\n";
    cout << "Recall:      " << metrics.recall * 100 << "%\n";
    cout << "Overhead:    " << metrics.overhead * 100 << "%\n";
    cout << "F1 Score:    " << metrics.f1 * 100 << "%\n";
    cout << "================================\n";
}

// ------------------------------------------------------------
// Create a timestamped run directory in Results/runs/
// ------------------------------------------------------------
string makeRunDirectory(const ExperimentConfig& experiment, int mpi_ranks)
{
    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    tm local_tm = *localtime(&now_time);
    auto millis = chrono::duration_cast<chrono::milliseconds>(
        now.time_since_epoch()
    ) % 1000;

    ostringstream oss;
    oss << "Results/runs/run_"
        << put_time(&local_tm, "%Y%m%d_%H%M%S")
        << "_" << setw(3) << setfill('0') << millis.count()
        << "_r" << mpi_ranks
        << "_seed" << experiment.splitSeed
        << "_t" << experiment.nTrees
        << "_d" << experiment.maxDepth
        << "_ms" << experiment.minSamplesSplit
        << "_mf" << experiment.maxFeatures
        << "_tr" << fixed << setprecision(2) << experiment.trainFraction;

    filesystem::create_directories(oss.str());
    return oss.str();
}

// ------------------------------------------------------------
// Save run metrics as JSON for downstream plotting
// ------------------------------------------------------------
void writeMetricsJson(
    const string& run_dir,
    const Metrics& metrics,
    int train_size,
    int validation_size,
    int mpi_ranks,
    double runtime_sec,
    const ExperimentConfig& experiment
)
{
    ofstream out(run_dir + "/metrics.json");
    out << "{\n";
    out << "  \"model_name\": \"Random Forest (Hybrid)\",\n";
    out << "  \"train_size\": " << train_size << ",\n";
    out << "  \"validation_size\": " << validation_size << ",\n";
    out << "  \"mpi_ranks\": " << mpi_ranks << ",\n";
    out << "  \"train_fraction\": " << experiment.trainFraction << ",\n";
    out << "  \"split_seed\": " << experiment.splitSeed << ",\n";
    out << "  \"n_trees\": " << experiment.nTrees << ",\n";
    out << "  \"max_depth\": " << experiment.maxDepth << ",\n";
    out << "  \"min_samples_split\": " << experiment.minSamplesSplit << ",\n";
    out << "  \"max_features\": " << experiment.maxFeatures << ",\n";
    out << "  \"runtime_sec\": " << runtime_sec << ",\n";
    out << "  \"accuracy\": " << metrics.accuracy * 100.0 << ",\n";
    out << "  \"precision\": " << metrics.precision * 100.0 << ",\n";
    out << "  \"recall\": " << metrics.recall * 100.0 << ",\n";
    out << "  \"f1\": " << metrics.f1 * 100.0 << ",\n";
    out << "  \"overhead\": " << metrics.overhead * 100.0 << ",\n";
    out << "  \"tp\": " << metrics.tp << ",\n";
    out << "  \"tn\": " << metrics.tn << ",\n";
    out << "  \"fp\": " << metrics.fp << ",\n";
    out << "  \"fn\": " << metrics.fn << "\n";
    out << "}\n";
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char* argv[])
{
    ExperimentConfig experiment;
    if (!parseArgs(argc, argv, experiment)) {
        return 1;
    }

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto t_start = chrono::high_resolution_clock::now();

    vector<vector<double>> X_raw;
    vector<int> y;

    if (rank == 0) {
        cout << "============================================================\n";
        cout << "Hybrid MPI + OpenMP Random Forest\n";
        cout << "============================================================\n";

        // Load dataset (adjust filename as needed)
        readCSV("diabetes.csv", X_raw, y);
        cout << "Loaded dataset: " << X_raw.size()
             << " samples, " << X_raw[0].size()
             << " features\n";
        cout << "Experiment config: train_fraction=" << experiment.trainFraction
             << ", split_seed=" << experiment.splitSeed
             << ", trees=" << experiment.nTrees
             << ", max_depth=" << experiment.maxDepth
             << ", min_samples_split=" << experiment.minSamplesSplit
             << ", max_features=" << experiment.maxFeatures << "\n";
    }

    // ------------------------------------------------------------
    // Broadcast dataset size to all ranks
    // ------------------------------------------------------------
    int n_samples = 0;
    int n_features = 0;

    if (rank == 0) {
        n_samples = X_raw.size();
        n_features = X_raw[0].size();
    }

    MPI_Bcast(&n_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_features, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // Broadcast labels
    // ------------------------------------------------------------
    if (rank != 0)
        y.resize(n_samples);

    MPI_Bcast(y.data(), n_samples, MPI_INT, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // Prepare storage for raw X if not rank 0
    // ------------------------------------------------------------
    vector<vector<double>> X_local;

    if (rank != 0) {
        X_local.assign(n_samples, vector<double>(n_features));
    } else {
        X_local = X_raw;
    }

    // ------------------------------------------------------------
    // Broadcast each column separately (contiguous per column)
    // ------------------------------------------------------------
    for (int j = 0; j < n_features; ++j) {
        vector<double> col(n_samples);

        if (rank == 0) {
            for (int i = 0; i < n_samples; ++i)
                col[i] = X_raw[i][j];
        }

        MPI_Bcast(col.data(), n_samples, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < n_samples; ++i)
            X_local[i][j] = col[i];
    }

    // ------------------------------------------------------------
    // Split data into train/validation subsets on all ranks
    // ------------------------------------------------------------
    vector<int> train_idx;
    vector<int> val_idx;
    stratifiedTrainValidationSplit(y, experiment.trainFraction, experiment.splitSeed, train_idx, val_idx);

    vector<vector<double>> X_train_raw;
    vector<vector<double>> X_val_raw;
    vector<int> y_train;
    vector<int> y_val;

    extractSubset(X_local, y, train_idx, X_train_raw, y_train);
    extractSubset(X_local, y, val_idx, X_val_raw, y_val);

    // Impute invalid zeros using training-only statistics to avoid leakage.
    vector<double> invalidZeroMedians = computeInvalidZeroMedians(X_train_raw, n_features);
    imputeInvalidZeros(X_train_raw, invalidZeroMedians);
    imputeInvalidZeros(X_val_raw, invalidZeroMedians);

    if (rank == 0) {
        cout << "Train/validation split: "
             << X_train_raw.size() << " train, "
             << X_val_raw.size() << " validation\n";
        int train_pos = static_cast<int>(count(y_train.begin(), y_train.end(), 1));
        int val_pos = static_cast<int>(count(y_val.begin(), y_val.end(), 1));
        cout << "Stratified class counts: train_pos=" << train_pos
             << "/" << y_train.size()
             << ", val_pos=" << val_pos
             << "/" << y_val.size() << "\n";
        cout << "Applied median imputation for zero-invalid columns: ";
        for (size_t i = 0; i < INVALID_ZERO_COLUMNS.size(); ++i) {
            int col = INVALID_ZERO_COLUMNS[i];
            cout << col << "->" << invalidZeroMedians[col];
            if (i + 1 < INVALID_ZERO_COLUMNS.size())
                cout << ", ";
        }
        cout << "\n";
    }

    // ------------------------------------------------------------
    // Build bin mappings on training data only and broadcast
    // ------------------------------------------------------------
    vector<BinMapping> mappings;

    if (rank != 0) {
        mappings.resize(n_features);
    }

    if (rank == 0) {
        mappings = computeBinMappings(X_train_raw);
    }

    for (int j = 0; j < n_features; j++) {
        MPI_Bcast(&mappings[j].minVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&mappings[j].maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&mappings[j].step,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // ------------------------------------------------------------
    // Convert train/validation datasets -> 256 bins (column-major)
    // ------------------------------------------------------------
    vector<vector<uint8_t>> X_train_bins;
    vector<vector<uint8_t>> X_val_bins;
    binarizeDataset(X_train_raw, mappings, X_train_bins);
    binarizeDataset(X_val_raw, mappings, X_val_bins);

    // ------------------------------------------------------------
    // Random Forest config - Optimized for 98% F1 score
    // ------------------------------------------------------------
    ForestConfig cfg;
    cfg.nTrees         = experiment.nTrees;
    cfg.maxDepth       = experiment.maxDepth;
    cfg.minSamplesSplit = experiment.minSamplesSplit;
    cfg.maxFeatures    = (experiment.maxFeatures == -1)
                       ? max(1, static_cast<int>(sqrt(static_cast<double>(n_features))))
                       : min(experiment.maxFeatures, n_features);
    cfg.numClasses     = 2;
    cfg.seed           = 42;    // Fixed seed for reproducibility
    experiment.maxFeatures = cfg.maxFeatures;

    if (rank == 0) {
        cout << "Training forest with " << cfg.nTrees
             << " trees using " << size
             << " MPI ranks and OpenMP threads...\n";
    }

    // ------------------------------------------------------------
    // Train MPI Random Forest
    // ------------------------------------------------------------
    MPIRandomForest forest(X_train_bins, y_train, cfg);
    forest.fit();

    // ------------------------------------------------------------
    // Predict using MPI reduction
    // ------------------------------------------------------------
    vector<int> preds = forest.predict(X_val_bins);

    // ------------------------------------------------------------
    // Only rank 0 prints metrics & HPC statistics
    // ------------------------------------------------------------
    if (rank == 0) {
        auto t_end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = t_end - t_start;
        double T_parallel = elapsed.count();
        Metrics metrics = computeMetrics(y_val, preds);

        evaluateMetrics(metrics);
        fullMetrics(metrics);

        // --------------------------------------------------------
        // Memory usage on macOS (ru_maxrss in bytes)
        // --------------------------------------------------------
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        double memMB = usage.ru_maxrss / (1024.0 * 1024.0);

        // --------------------------------------------------------
        // Print parallel timing & memory
        // --------------------------------------------------------
        cout << "\n---------------------------------------------------\n";
        cout << "T_parallel (seconds): " << T_parallel << "\n";
        cout << "Memory Usage (MB):    " << memMB << "\n";
        cout << "MPI Ranks:            " << size << "\n";
        cout << "---------------------------------------------------\n";

        string run_dir = makeRunDirectory(experiment, size);
        writeMetricsJson(
            run_dir,
            metrics,
            static_cast<int>(X_train_raw.size()),
            static_cast<int>(X_val_raw.size()),
            size,
            T_parallel,
            experiment
        );
        cout << "Saved metrics to:     " << run_dir << "/metrics.json\n";
    }

    MPI_Finalize();
    return 0;
}
