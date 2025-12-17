
void execute_hybrid() {
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
    // Build bin mappings on rank 0 and broadcast
    // ------------------------------------------------------------
    vector<BinMapping> mappings;

    if (rank != 0) {
        mappings.resize(n_features);
    }

    if (rank == 0) {
        mappings = computeBinMappings(X_local);
    }

    for (int j = 0; j < n_features; j++) {
        MPI_Bcast(&mappings[j].minVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&mappings[j].maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&mappings[j].step,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // ------------------------------------------------------------
    // Convert dataset -> 256 bins (column-major)
    // ------------------------------------------------------------
    vector<vector<uint8_t>> X_bins;
    binarizeDataset(X_local, mappings, X_bins);

    // ------------------------------------------------------------
    // Random Forest config - Optimized for 98% F1 score
    // ------------------------------------------------------------
    ForestConfig cfg;
    cfg.nTrees         = 500;   // Significantly increased for maximum accuracy
    cfg.maxDepth       = 20;    // Deeper trees for better feature learning
    cfg.minSamplesSplit = 5;   // Lower threshold to allow more splits
    cfg.maxFeatures    = n_features;  // Use all features for maximum information
    cfg.numClasses     = 2;
    cfg.seed           = 42;    // Fixed seed for reproducibility

    if (rank == 0) {
        cout << "Training forest with " << cfg.nTrees
             << " trees using " << size
             << " MPI ranks and OpenMP threads...\n";
    }

    // ------------------------------------------------------------
    // Train MPI Random Forest
    // ------------------------------------------------------------
    MPIRandomForest forest(X_bins, y, cfg);
    forest.fit();

    // ------------------------------------------------------------
    // Predict using MPI reduction
    // ------------------------------------------------------------
    vector<int> preds = forest.predict();

    // ------------------------------------------------------------
    // Only rank 0 prints metrics & HPC statistics
    // ------------------------------------------------------------
    if (rank == 0) {
        auto t_end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = t_end - t_start;
        double T_parallel = elapsed.count();

        evaluateMetrics(y, preds);
        fullMetrics(y, preds);

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
    }

    MPI_Finalize();
    return 0;
}