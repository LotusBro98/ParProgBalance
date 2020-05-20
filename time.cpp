#include <mpi.h>
#include <cstdlib>

#define N_RUNS 1000
#define BUF_SIZE 1024

int main(int argc, char* argv[]) {
    MPI::Init(argc, argv);
    int size = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();
    double time_start, time_end;

    char * buf = (char*)malloc(BUF_SIZE);
    char * buf2 = (char*)malloc(BUF_SIZE);

    time_start = MPI::Wtime();
    for (int i = 0; i < N_RUNS; i++) {
        MPI::COMM_WORLD.Bcast(buf, BUF_SIZE, MPI::CHAR, 0);
    }
    time_end = MPI::Wtime();
    if (rank == 0) {
        printf("MPI_Bcast: %lg\n", (time_end - time_start) / N_RUNS);
    }

    time_start = MPI::Wtime();
    for (int i = 0; i < N_RUNS; i++) {
        MPI::COMM_WORLD.Reduce(buf2, buf, BUF_SIZE, MPI::CHAR, MPI::SUM, 0);
    }
    time_end = MPI::Wtime();
    if (rank == 0) {
        printf("MPI_Reduce: %lg\n", (time_end - time_start) / N_RUNS);
    }

    time_start = MPI::Wtime();
    for (int i = 0; i < N_RUNS; i++) {
        MPI::COMM_WORLD.Gather(buf, BUF_SIZE / size, MPI::CHAR, buf2, BUF_SIZE / size, MPI::CHAR, 0);
    }
    time_end = MPI::Wtime();
    if (rank == 0) {
        printf("MPI_Gather: %lg\n", (time_end - time_start) / N_RUNS);
    }

    time_start = MPI::Wtime();
    for (int i = 0; i < N_RUNS; i++) {
        MPI::COMM_WORLD.Scatter(buf, BUF_SIZE / size, MPI::CHAR, buf2, BUF_SIZE / size, MPI::CHAR, 0);
    }
    time_end = MPI::Wtime();
    if (rank == 0) {
        printf("MPI_Scatter: %lg\n", (time_end - time_start) / N_RUNS);
    }

    MPI::COMM_WORLD.Barrier();
}