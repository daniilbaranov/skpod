#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <ctime>
#include <sstream>

using namespace std;

const double EPS = 1E-9;

int main(int argc, char **argv) {
    int rang, resRang = 0, curProc, curPos, imax, maxPos;
    double resTime = 0, max, element;
    int nRows;
    sscanf(argv[1], "%d", &nRows);
    int nProcs, rank, i, j, k, d, v, p;
    MPI_Status status;
    double rt, t1, t2;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    struct {
        double value;
        int   index;
    } individual, common;
    rang = nRows;
    int N = nRows / nProcs;
    double *a = new double[nRows * N];
    double mul;
    double V[nRows];
    bool used[N];
    
    for(i = 0; i < N; i++) {
        used[i] = false;
        for(j = 0; j < nRows; j++) {
            a[i * nRows + j] = (i + N * rank == j) ? 1 : 2;
        }
    }

    t1 = MPI_Wtime();
    
    for (i = 0; i < nRows; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        max = 0;
        maxPos = 0;
        for (j = 0; j < N; ++j) {
            if (!used[j]) {
                element = fabs(a[j * nRows + i]);
                if (element > max) {
                    max = element;
                    maxPos = j;
                }
            }
        }
        
        individual.value = (max < EPS) ? 0 : max;
        individual.index = maxPos + rank * N;
        
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&individual, &common, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        if (common.value == 0) {
            --rang;
            continue;
        }
        
        if (i == nRows - 1)
            break;
        
        curPos = common.index % N;
        curProc = common.index / N;
        
        if (curProc == rank) {
            for (d = 0; d < nProcs; ++d) {
                if (d != curProc)
                    MPI_Send(&a[curPos * nRows], nRows, MPI_DOUBLE, d, 1, MPI_COMM_WORLD);
            }
            used[curPos] = true;
            for (j = 0; j < N; ++j) {
                if (!used[j] && fabs(a[j * nRows + i]) > 0) {
                    mul = -a[j * nRows + i] / a[curPos * nRows + i];
                    for (k = i; k < nRows; ++k) {
                        a[j * nRows + k] += a[curPos * nRows + k] * mul;
                    }
                }
            }
        } else {
            MPI_Recv(V, nRows, MPI_DOUBLE, curProc, 1, MPI_COMM_WORLD, &status);
            for (j = 0; j < N; ++j) {
                if (!used[j] && fabs(a[j * nRows + i]) > 0) {
                    mul = -a[j * nRows + i] / V[i];
                    for (k = i; k < nRows; ++k) {
                        a[j * nRows + k] += V[k] * mul;
                    }
                }
            }
        }
    }
    t2 = MPI_Wtime();
    rt = t2 - t1;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&rt, &resTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        cout << "Rang: " << rang << " Time: " << resTime << endl;
    }
    delete[] a;
    MPI_Finalize();
    return 0;
}
