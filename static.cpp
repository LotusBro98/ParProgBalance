#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string.h>
#include <unistd.h>

class Block {
public:
    Block(std::vector<int> &bufferA, std::vector<int> &bufferB, int start, int end):
    carry0(0),
    carry1(1),
    bufferA(bufferA.begin() + start, bufferA.begin() + end),
    bufferB(bufferB.begin() + start, bufferB.begin() + end)
    {
    }

    // Receive block from root
    explicit Block(int rank) : carry0(0), carry1(1)
    {
        int NA, NB;
        MPI::COMM_WORLD.Recv(&NA, 1, MPI::INT, 0, 0);
        MPI::COMM_WORLD.Recv(&NB, 1, MPI::INT, 0, 0);
        bufferA.resize(NA);
        bufferB.resize(NB);
        MPI::COMM_WORLD.Recv(bufferA.data(), NA, MPI::INT, 0, 0);
        MPI::COMM_WORLD.Recv(bufferB.data(), NB, MPI::INT, 0, 0);

        int N = std::max(bufferA.size(), bufferB.size());
        result0.resize(N);
        result1.resize(N);
    }

    void transfer(int rank) {
        int NA = bufferA.size();
        int NB = bufferB.size();
        MPI::COMM_WORLD.Send(&NA, 1, MPI::INT, rank, 0);
        MPI::COMM_WORLD.Send(&NB, 1, MPI::INT, rank, 0);
        MPI::COMM_WORLD.Send(bufferA.data(), NA, MPI::INT, rank, 0);
        MPI::COMM_WORLD.Send(bufferB.data(), NB, MPI::INT, rank, 0);
    }

    void process() {
        int N = std::max(bufferA.size(), bufferB.size());
        for (int i = 0; i < N; i++)
        {
            int sum0 = bufferA[i] + bufferB[i] + carry0;
            int sum1 = bufferA[i] + bufferB[i] + carry1;
            carry0 = sum0 / 1'000'000'000;
            carry1 = sum1 / 1'000'000'000;
            result0[i] = sum0 % 1'000'000'000;
            result1[i] = sum1 % 1'000'000'000;
        }
    }

    void transfer_result() {
        int N = std::max(bufferA.size(), bufferB.size());
        int carry;
        MPI::COMM_WORLD.Recv(&carry, 1, MPI::INT, 0, 0);
        void* buf;
        if (carry == 0) {
            buf = result0.data();
            carry = carry0;
        } else {
            buf = result1.data();
            carry = carry1;
        }
        MPI::COMM_WORLD.Send(&carry, 1, MPI::INT, 0, 0);
        MPI::COMM_WORLD.Send(buf, N, MPI::INT, 0, 0);
    }

    void collect_result(int rank, void* buf, int &carry) {
        int N = std::max(bufferA.size(), bufferB.size());
        MPI::COMM_WORLD.Send(&carry, 1, MPI::INT, rank, 0);
        MPI::COMM_WORLD.Recv(&carry, 1, MPI::INT, rank, 0);
        MPI::COMM_WORLD.Recv(buf, N, MPI::INT, rank, 0);
    }

    friend std::ostream& operator <<(std::ostream& os, Block* block) {
        for (int i = block->bufferA.size() - 1; i >= 0 ; i--)
            std::cout << std::setw(9) << std::setfill('0') << block->bufferA[i] << "'";
        std::cout << "\n";
        for (int i = block->bufferB.size() - 1; i >= 0 ; i--)
            std::cout << std::setw(9) << std::setfill('0') << block->bufferB[i] << "'";
        std::cout << "\n";
        return os;
    }

    std::vector<int> result0;
    std::vector<int> result1;
    int carry0;
    int carry1;

private:
    std::vector<int> bufferA;
    std::vector<int> bufferB;
};

void readFile(int &lenA, int &lenB, std::vector<int> &bufferA, std::vector<int> &bufferB, char* filename)
{
    FILE* file = fopen(filename, "rt");
    fscanf(file, "%d %d", &lenA, &lenB);
    int NA = lenA / 9 + ((lenA % 9) != 0);
    int NB = lenB / 9 + ((lenB % 9) != 0);
    bufferA.resize(NA);
    bufferB.resize(NB);
    char buf[10] = {};
    fgetc(file); // \n
    for (int i = NA - 1; i >= 0; i--) {
        int n = ((lenA % 9 != 0) && (i == NA - 1)) ? lenA % 9 : 9;
        int code = fread(buf, 1, n, file);
        bufferA[i] = atoi(buf);
    }
    fgetc(file); // \n
    for (int i = NB - 1; i >= 0; i--) {
        int n = ((lenB % 9 != 0) && (i == NB - 1)) ? lenB % 9 : 9;
        int code = fread(buf, 1, n, file);
        bufferB[i] = atoi(buf);
    }
}

int rootMain(int size, char* filename)
{
    int lenA, lenB;
    std::vector<int> bufferA;
    std::vector<int> bufferB;

    readFile(lenA, lenB, bufferA, bufferB, filename);
    int N = std::max(bufferA.size(), bufferB.size());
    bufferA.resize(N, 0);
    bufferB.resize(N, 0);

    auto start_time = MPI::Wtime();

    int n_slaves = size - 1;
    int chunk = (N % n_slaves == 0) ? N / n_slaves : N / n_slaves + 1;
    Block** blocks = new Block*[n_slaves];
    for (int slave = 0; slave < n_slaves; slave++) {
        int start = chunk * (slave);
        int end = std::min(chunk * (slave + 1), N);
        blocks[slave] = new Block(bufferA, bufferB, start, end);
        int rank = slave + 1;
        blocks[slave]->transfer(rank);
    }

    std::vector<int> result;
    result.resize(N);

    int carry = 0;
    for (int slave = 0; slave < n_slaves; slave++) {
        int rank = slave + 1;
        int start = chunk * (slave);
        blocks[slave]->collect_result(rank, result.data() + start, carry);
    }
    // Result is ready
    auto end_time = MPI::Wtime();

    std::cout << end_time - start_time << "\n";

    std::ofstream ofile("output");
    if (carry == 1)
        ofile << "1";

    for (int i = N - 1; i >= 0; i--) {
        if (i == N - 1 && carry == 0)
            ofile << result[i];
        else
            ofile << std::setw(9) << std::setfill('0') << result[i];
    }
    ofile << "\n";
    ofile.close();

    MPI::COMM_WORLD.Barrier();

    return 0;
}

int slaveMain(int rank, int size)
{
    auto * block = new Block(rank);

    block->process();
    block->transfer_result();

    MPI::COMM_WORLD.Barrier();

    return 0;
}

int main(int argc, char* argv[]) {
    MPI::Init(argc, argv);
    int size = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();

    if (rank == 0) {
        if (argc < 2) {
            std::cout << "Usage: mpirun -n <n_processes> ./static_balance <input_filename>\n";
            return -1;
        }
        char* filename = argv[1];
        return rootMain(size, filename);
    }
    else {
        return slaveMain(rank, size);
    }
}
