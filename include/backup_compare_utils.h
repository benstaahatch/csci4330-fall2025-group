#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <vector>

using namespace std;

struct CompareMetrics {
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

struct CompareExperimentConfig {
    double trainFraction = 0.8;
    int splitSeed = 42;
    int nTrees = 100;
    int maxDepth = 10;
    int minSamplesSplit = 2;
    int maxFeatures = -1;
};

inline const vector<int> BACKUP_INVALID_ZERO_COLUMNS = {1, 2, 3, 4, 5};

inline double getMemoryUsageMB()
{
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

inline void printCompareUsage(const char* programName, bool include_threads)
{
    cout << "Usage: " << programName
         << " [--train-fraction FLOAT] [--split-seed INT] [--trees INT]"
         << " [--max-depth INT] [--min-samples-split INT] [--max-features INT]";
    if (include_threads) {
        cout << " [--threads INT]";
    }
    cout << "\n";
}

inline bool parseCompareArgs(
    int argc,
    char* argv[],
    CompareExperimentConfig& config,
    int* omp_threads = nullptr
)
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
        } else if (arg == "--threads") {
            if (!omp_threads) {
                cerr << "--threads is only supported by the OpenMP comparison binary.\n";
                return false;
            }
            if (!requireValue(arg)) return false;
            *omp_threads = stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printCompareUsage(argv[0], omp_threads != nullptr);
            return false;
        } else {
            cerr << "Unknown argument: " << arg << "\n";
            printCompareUsage(argv[0], omp_threads != nullptr);
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
    if (omp_threads && *omp_threads <= 0) {
        cerr << "--threads must be positive.\n";
        return false;
    }

    return true;
}

inline void readCSV(const string& filename, vector<vector<double>>& X, vector<int>& y)
{
    ifstream file(filename);
    if (!file) {
        filesystem::path alt_path = filesystem::path("Data") / filename;
        file.open(alt_path);
    }
    if (!file) {
        throw runtime_error("Failed to open dataset: " + filename + " or Data/" + filename);
    }

    string line;
    bool skipHeader = true;
    while (getline(file, line)) {
        if (skipHeader) {
            skipHeader = false;
            continue;
        }

        stringstream ss(line);
        string value;
        vector<double> row;
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }

        y.push_back(static_cast<int>(row.back()));
        row.pop_back();
        X.push_back(row);
    }
}

inline void stratifiedTrainValidationSplit(
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

    for (int i = 0; i < static_cast<int>(y.size()); ++i) {
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

inline void extractSubset(
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

inline vector<double> computeInvalidZeroMedians(
    const vector<vector<double>>& X_train,
    int n_features
)
{
    vector<double> medians(n_features, 0.0);

    for (int col : BACKUP_INVALID_ZERO_COLUMNS) {
        vector<double> values;
        values.reserve(X_train.size());

        for (const auto& row : X_train) {
            if (row[col] != 0.0) {
                values.push_back(row[col]);
            }
        }

        if (values.empty()) {
            continue;
        }

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

inline void imputeInvalidZeros(vector<vector<double>>& X, const vector<double>& medians)
{
    for (auto& row : X) {
        for (int col : BACKUP_INVALID_ZERO_COLUMNS) {
            if (row[col] == 0.0 && medians[col] > 0.0) {
                row[col] = medians[col];
            }
        }
    }
}

struct MinMaxStats {
    vector<double> minVals;
    vector<double> maxVals;
};

inline MinMaxStats computeMinMaxStats(const vector<vector<double>>& X_train)
{
    int n_features = static_cast<int>(X_train[0].size());
    MinMaxStats stats;
    stats.minVals.assign(n_features, X_train[0][0]);
    stats.maxVals.assign(n_features, X_train[0][0]);

    for (int j = 0; j < n_features; ++j) {
        stats.minVals[j] = X_train[0][j];
        stats.maxVals[j] = X_train[0][j];
        for (size_t i = 1; i < X_train.size(); ++i) {
            stats.minVals[j] = min(stats.minVals[j], X_train[i][j]);
            stats.maxVals[j] = max(stats.maxVals[j], X_train[i][j]);
        }
    }

    return stats;
}

inline void applyMinMaxScale(vector<vector<double>>& X, const MinMaxStats& stats)
{
    int n_features = static_cast<int>(stats.minVals.size());
    for (auto& row : X) {
        for (int j = 0; j < n_features; ++j) {
            double minVal = stats.minVals[j];
            double maxVal = stats.maxVals[j];
            row[j] = (maxVal != minVal) ? ((row[j] - minVal) / (maxVal - minVal)) : 0.0;
        }
    }
}

inline CompareMetrics computeMetrics(const vector<int>& y_true, const vector<int>& y_pred)
{
    CompareMetrics metrics;

    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_pred[i] == 1 && y_true[i] == 1) metrics.tp++;
        else if (y_pred[i] == 0 && y_true[i] == 0) metrics.tn++;
        else if (y_pred[i] == 1 && y_true[i] == 0) metrics.fp++;
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

inline void printMetrics(const CompareMetrics& metrics)
{
    cout << "\n---- Metrics ----\n";
    cout << "TP: " << metrics.tp << "  TN: " << metrics.tn << "\n";
    cout << "FP: " << metrics.fp << "  FN: " << metrics.fn << "\n";
    cout << "Accuracy: " << metrics.accuracy * 100.0 << "%\n";
    cout << "\n===== FULL MODEL METRICS =====\n";
    cout << "Accuracy:    " << metrics.accuracy * 100.0 << "%\n";
    cout << "Precision:   " << metrics.precision * 100.0 << "%\n";
    cout << "Recall:      " << metrics.recall * 100.0 << "%\n";
    cout << "Overhead:    " << metrics.overhead * 100.0 << "%\n";
    cout << "F1 Score:    " << metrics.f1 * 100.0 << "%\n";
    cout << "================================\n";
}

inline string makeRunDirectory(
    const CompareExperimentConfig& experiment,
    const string& model_tag,
    int mpi_ranks,
    int omp_threads
)
{
    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    tm local_tm = *localtime(&now_time);
    auto millis = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;

    ostringstream oss;
    oss << "Results/runs/run_"
        << put_time(&local_tm, "%Y%m%d_%H%M%S")
        << "_" << setw(3) << setfill('0') << millis.count()
        << "_" << model_tag
        << "_r" << mpi_ranks
        << "_th" << omp_threads
        << "_seed" << experiment.splitSeed
        << "_t" << experiment.nTrees
        << "_d" << experiment.maxDepth
        << "_ms" << experiment.minSamplesSplit
        << "_mf" << experiment.maxFeatures
        << "_tr" << fixed << setprecision(2) << experiment.trainFraction;

    filesystem::create_directories(oss.str());
    return oss.str();
}

inline void writeMetricsJson(
    const string& run_dir,
    const string& model_name,
    const CompareMetrics& metrics,
    int train_size,
    int validation_size,
    int mpi_ranks,
    int omp_threads,
    double runtime_sec,
    const CompareExperimentConfig& experiment
)
{
    ofstream out(run_dir + "/metrics.json");
    out << "{\n";
    out << "  \"model_name\": \"" << model_name << "\",\n";
    out << "  \"train_size\": " << train_size << ",\n";
    out << "  \"validation_size\": " << validation_size << ",\n";
    out << "  \"mpi_ranks\": " << mpi_ranks << ",\n";
    out << "  \"omp_threads\": " << omp_threads << ",\n";
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
