#include <mpi.h>

#define WAY_1

int main(int argc, char* argv[]) {
    MPI::Init(argc, argv);
    int size = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();

#ifdef WAY_1
    int buf;

    if (rank != 0) {
        MPI::COMM_WORLD.Recv(&buf, 1, MPI::INT, rank - 1, 0);
    }

    printf("%d ", rank);

    if (rank != size - 1) {
        MPI::COMM_WORLD.Send(&buf, 1, MPI::INT, rank + 1, 0);
    } else {
        printf("\n");
    }

    MPI::COMM_WORLD.Barrier();
#else
    int buf;

    if (rank == 0) {
        printf("%d ", rank);
        for (int  i = 1; i < size; i++) {
            MPI::COMM_WORLD.Send(&buf, 1, MPI::INT, i, 0);
            MPI::COMM_WORLD.Recv(&buf, 1, MPI::INT, i, 0);
        }
        printf("\n");
    } else {
        MPI::COMM_WORLD.Recv(&buf, 1, MPI::INT, 0, 0);
        printf("%d ", rank);
        MPI::COMM_WORLD.Send(&buf, 1, MPI::INT, 0, 0);
    }

    MPI::COMM_WORLD.Barrier();
#endif
}