#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <sys/resource.h>   // For memory usage
#include <omp.h>            // For OpenMP thread count
#include "bins.h"
#include "mpi_forest.h"
using namespace std;

/**
 * TODO: should be in a separate file -
 */
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
// Simple confusion matrix & accuracy
// ------------------------------------------------------------
void evaluateMetrics(const vector<int>& y, const vector<int>& pred)
{
    int TP=0, TN=0, FP=0, FN=0;

    for (int i = 0; i < y.size(); ++i) {
        if (pred[i] == 1 && y[i] == 1) TP++;
        else if (pred[i] == 0 && y[i] == 0) TN++;
        else if (pred[i] == 1 && y[i] == 0) FP++;
        else FN++;
    }

    double acc = (double)(TP + TN) / (TP + TN + FP + FN);

    cout << "\n---- Metrics ----\n";
    cout << "TP: " << TP << "  TN: " << TN << "\n";
    cout << "FP: " << FP << "  FN: " << FN << "\n";
    cout << "Accuracy: " << acc * 100 << "%\n";
}

// ------------------------------------------------------------
// Full metrics including Overhead instead of Specificity
// ------------------------------------------------------------
void fullMetrics(const vector<int>& y, const vector<int>& pred)
{
    int TP=0, TN=0, FP=0, FN=0;

    for (int i = 0; i < y.size(); i++) {
        if (pred[i] == 1 && y[i] == 1) TP++;
        else if (pred[i] == 0 && y[i] == 0) TN++;
        else if (pred[i] == 1 && y[i] == 0) FP++;
        else FN++;
    }

    double accuracy    = (double)(TP + TN) / (TP + TN + FP + FN);
    double precision   = (TP + FP == 0) ? 0 : (double)TP / (TP + FP);
    double recall      = (TP + FN == 0) ? 0 : (double)TP / (TP + FN);
    double f1          = (precision + recall == 0) ? 0 :
                         2.0 * (precision * recall) / (precision + recall);

    // Overhead (ML-related definition): extra misclassification
    double overheadML = 1.0 - accuracy;

    cout << "\n===== FULL MODEL METRICS =====\n";
    cout << "Accuracy:    " << accuracy * 100 << "%\n";
    cout << "Precision:   " << precision * 100 << "%\n";
    cout << "Recall:      " << recall * 100 << "%\n";
    cout << "Overhead:    " << overheadML * 100 << "%\n";
    cout << "F1 Score:    " << f1 * 100 << "%\n";
    cout << "================================\n";
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char* argv[])
{
    if(hybrid ...)
    if(serial ...)
    if(parallel ...)
}
