
/**
 * TODO: replace all "readCSV" calls with this
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