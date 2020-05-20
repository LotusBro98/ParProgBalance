#include <mpi.h>
#include <cmath>

struct Task {
    int from;
    int to;
    int offset;
    int step;
};

double f(int n)  {
    return 6 / (M_PI * M_PI * n * n);
}

double calc_sum(int from, int to, int offset, int step) {
    double sum = 0;
    for (int i = to - offset; i >= from; i -= step) {
        sum += f(i);
    }
    return sum;
}

int main(int argc, char* argv[]) {
    MPI::Init(argc, argv);
    int size = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();

    if (rank == 0) {
        if (argc != 2) {
            printf("Usage: mpirun -n <N_pr> ./sum <n>\n");
            return -1;
        }
        int n = atoi(argv[1]);

        double start_time = MPI::Wtime();
        struct Task tasks[size];
        for (int i = 1; i < size; i++) {
            tasks[i] = {1, n,i - 1, size - 1};
        }
        MPI::COMM_WORLD.Scatter(tasks, sizeof(struct Task), MPI::CHAR, nullptr, 0, MPI::CHAR, 0);
        double sum = 0;
        double all_sum;
        MPI::COMM_WORLD.Reduce(&sum, &all_sum, 1, MPI::DOUBLE, MPI::SUM, 0);
        double end_time = MPI::Wtime();
        printf("%.15lg\n", all_sum);
        fprintf(stderr, "calc_time: %lf\n", end_time - start_time);
    } else {
        struct Task task;
        MPI::COMM_WORLD.Scatter(nullptr, 0, MPI::CHAR, &task, sizeof(struct Task), MPI::CHAR, 0);
        double sum = calc_sum(task.from, task.to, task.offset, task.step);
        double all_sum;
        MPI::COMM_WORLD.Reduce(&sum, &all_sum, 1, MPI::DOUBLE, MPI::SUM, 0);
    }

    MPI::COMM_WORLD.Barrier();
}