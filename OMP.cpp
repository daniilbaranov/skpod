#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <sstream>
#include <iomanip>
#include <assert.h>

using namespace std;

const double EPS = 1E-9;

int colMaxOMP(const vector <vector <double> > &matrix, int col, int n) {
    double max = fabs(matrix[col][col]);
    int maxPos = col;
#pragma omp parallel
{
    double loc_max = max;
    int loc_max_pos = maxPos;
#pragma omp for
    for (int i = col + 1; i < n; ++i) {
        double element = fabs(matrix[i][col]);
        if (element > loc_max) {
            loc_max = element;
            loc_max_pos = i;
        }
    }
#pragma omp critical
{
   if (max < loc_max) {
        max = loc_max;
        maxPos = loc_max_pos;
   }
}
}
    return (max < EPS) ? -1 : maxPos;
}

int rankOMP(vector <vector <double> > &matrix, int n) {
    int rank = n;
    for (int i = 0; i < n; ++i) {
        int imax = colMaxOMP(matrix, i, n);
        if (imax == -1) {
            --rank;
            continue;
        }
        if (i != imax) {
            swap(matrix[i], matrix[imax]);
        }
#pragma omp parallel for
        for (int j = i + 1; j < n; ++j) {
          if (fabs(matrix[j][i]) > 0) {
            double mul = -matrix[j][i] / matrix[i][i];
            for (int k = i; k < n; ++k) {
                matrix[j][k] += matrix[i][k] * mul;
            }
          }
        }
    }
    return rank;
}

int colMaxBasic(const vector <vector <double> > &matrix, int col, int n) {
    double max = fabs(matrix[col][col]);
    int maxPos = col;
    for (int i = col + 1; i < n; ++i) {
        double element = fabs(matrix[i][col]);
        if (element > max) {
            max = element;
            maxPos = i;
        }
    }
    return (max < EPS) ? -1 : maxPos;
}

int rankBasic(vector <vector <double> > &matrix, int n) {
    int rank = n;
    for (int i = 0; i < n; ++i) {
        int imax = colMaxBasic(matrix, i, n);
        if (imax == -1) {
            --rank;
            continue;
        }
        if (i != imax) {
            swap(matrix[i], matrix[imax]);
        }
        for (int j = i + 1; j < n; ++j) {
          if (fabs(matrix[j][i] > 0)) {
            double mul = -matrix[j][i] / matrix[i][i];
            for (int k = i; k < n; ++k) {
                matrix[j][k] += matrix[i][k] * mul;
            }
          }
        }
    }
    return rank;
}

void printMatrix(vector <vector <double> > &matrix, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(matrix[i][j]) < EPS)
                cout << 0 << "\t";
            else
                cout << matrix[i][j] << "\t";
        }
        cout << endl;
    }
}

int main(int argc, char *argv[]) {

    int nRows, nThreads;

    sscanf(argv[1], "%d", &nRows);

    vector <vector <double> > a(nRows);
    vector <vector <double> > b(nRows);

    for (int i = 0; i < nRows; ++i) {
        a[i].resize(nRows);
        b[i].resize(nRows);
        for (int j = 0; j < nRows; ++j)
            b[i][j] = a[i][j] = rand();
    }
    double timerOpenMP = omp_get_wtime();
    int ROMP = rankOMP(a, nRows);
    timerOpenMP = omp_get_wtime() - timerOpenMP;

    cout << fixed << setprecision(10);
    cout << timerOpenMP << "\t\t" << "// OpenMP. nTreads: " << getenv("OMP_NUM_THREADS") << " nRows: " << nRows << endl;
    return 0;
}

