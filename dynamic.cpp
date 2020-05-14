#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string.h>
#include <unistd.h>

enum {CMD_WORK, CMD_EXIT};
enum {TAG_DATA, TAG_CMD, TAG_READY};

class Block {
public:
    Block(std::vector<int> &bufferA, std::vector<int> &bufferB, int start, int end):
    carry0(0),
    carry1(1),
    bufferA(bufferA.begin() + start, bufferA.begin() + end),
    bufferB(bufferB.begin() + start, bufferB.begin() + end)
    {}

    Block() : carry0(0), carry1(1) {}

    // Receive block from root
    void receive() {
        int NA, NB;
        MPI::COMM_WORLD.Recv(&NA, 1, MPI::INT, 0, TAG_DATA);
        MPI::COMM_WORLD.Recv(&NB, 1, MPI::INT, 0, TAG_DATA);
        bufferA.resize(NA);
        bufferB.resize(NB);
        MPI::COMM_WORLD.Recv(bufferA.data(), NA, MPI::INT, 0, TAG_DATA);
        MPI::COMM_WORLD.Recv(bufferB.data(), NB, MPI::INT, 0, TAG_DATA);

        int N = std::max(bufferA.size(), bufferB.size());
        result0.resize(N);
        result1.resize(N);
    }

    void transfer(int rank) {
        int NA = bufferA.size();
        int NB = bufferB.size();
        MPI::COMM_WORLD.Send(&NA, 1, MPI::INT, rank, TAG_DATA);
        MPI::COMM_WORLD.Send(&NB, 1, MPI::INT, rank, TAG_DATA);
        MPI::COMM_WORLD.Send(bufferA.data(), NA, MPI::INT, rank, TAG_DATA);
        MPI::COMM_WORLD.Send(bufferB.data(), NB, MPI::INT, rank, TAG_DATA);
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

    void transfer_result(int rank) {
        MPI::COMM_WORLD.Send(&rank, 1, MPI::INT, 0, TAG_READY);
        int N = std::max(bufferA.size(), bufferB.size());
        MPI::COMM_WORLD.Send(&carry0, 1, MPI::INT, 0, TAG_DATA);
        MPI::COMM_WORLD.Send(&carry1, 1, MPI::INT, 0, TAG_DATA);
        MPI::COMM_WORLD.Send(result0.data(), N, MPI::INT, 0, TAG_DATA);
        MPI::COMM_WORLD.Send(result1.data(), N, MPI::INT, 0, TAG_DATA);
        done = true;
    }

    void collect_result(int rank) {
        int N = std::max(bufferA.size(), bufferB.size());
        result0.resize(N);
        result1.resize(N);
        MPI::COMM_WORLD.Recv(&carry0, 1, MPI::INT, rank, TAG_DATA);
        MPI::COMM_WORLD.Recv(&carry1, 1, MPI::INT, rank, TAG_DATA);
        MPI::COMM_WORLD.Recv(result0.data(), N, MPI::INT, rank, TAG_DATA);
        MPI::COMM_WORLD.Recv(result1.data(), N, MPI::INT, rank, TAG_DATA);
        done = true;
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
    bool done = false;
    int real_carry;

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
    int chunk = 1000'000;
    if (chunk > N / n_slaves) chunk = N / n_slaves;
    if (chunk < 1) chunk = 1;
    int n_chunks = N % chunk == 0 ? N / chunk : N / chunk + 1;
    Block** blocks = new Block*[n_chunks];
    for (int c = 0; c < n_chunks; c++) {
        int start = chunk * c;
        int end = std::min(chunk * (c + 1), N);
        blocks[c] = new Block(bufferA, bufferB, start, end);
    }

    int slaves_work[n_slaves];
    for (int slave = 0; slave < n_slaves; slave++) {
        int rank = slave + 1;
        int cmd = CMD_WORK;
        MPI::COMM_WORLD.Send(&cmd, 1, MPI::INT, rank, TAG_CMD);
        blocks[slave]->transfer(rank);
        blocks[slave]->done = true;
        slaves_work[slave] = slave;
    }

    for (int done_chunks = 0; done_chunks < n_chunks; done_chunks++) {
        int rank;
        MPI::COMM_WORLD.Recv(&rank, 1, MPI::INT, MPI::ANY_SOURCE, TAG_READY);
        int slave = rank - 1;
        int c = slaves_work[slave];
        blocks[c]->collect_result(rank);

        int undone = 0;
        for (;undone < n_chunks; undone++)
            if (!blocks[undone]->done)
                break;

        if (undone < n_chunks) {
            c = undone;
            slaves_work[slave] = c;
            int cmd = CMD_WORK;
            MPI::COMM_WORLD.Send(&cmd, 1, MPI::INT, rank, TAG_CMD);
            blocks[c]->transfer(rank);
            blocks[c]->done = true;
        }
    }

    int cmd = CMD_EXIT;
    for (int rank = 1; rank < size; rank++)
        MPI::COMM_WORLD.Send(&cmd, 1, MPI::INT, rank, TAG_CMD);

    auto end_time = MPI::Wtime();

    std::cout << end_time - start_time << "\n";

    std::ofstream ofile("output");

    int carry = 0;
    for (int i = 0; i < n_chunks; i++) {
        blocks[i]->real_carry = carry;
        carry = (carry == 0 ? blocks[i]->carry0 : blocks[i]->carry1);
    }

    if (carry == 1)
        ofile << "1";

    for (int i = N - 1; i >= 0; i--) {
        int c = i / chunk;
        int j = i % chunk;
        int num = (blocks[c]->real_carry == 0) ? blocks[c]->result0[j] : blocks[c]->result1[j];
        if (i == N - 1 && carry == 0)
            ofile << num;
        else
            ofile << std::setw(9) << std::setfill('0') << num;
    }
    ofile << "\n";
    ofile.close();

    MPI::COMM_WORLD.Barrier();

    return 0;
}

int slaveMain(int rank, int size)
{
    auto * block = new Block();

    while (true) {
        int cmd;
        MPI::COMM_WORLD.Recv(&cmd, 1, MPI::INT, 0, TAG_CMD);
        if (cmd == CMD_EXIT)
            break;
        else if (cmd == CMD_WORK) {
            block->receive();
            block->process();
            block->transfer_result(rank);
        }
    }

    MPI::COMM_WORLD.Barrier();

    return 0;
}

int main(int argc, char* argv[]) {
    MPI::Init(argc, argv);
    int size = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();

    if (rank == 0) {
        if (argc < 2) {
            std::cout << "Usage: mpirun -n <n_processes> ./dynamic_balance <input_filename>\n";
            return -1;
        }
        char* filename = argv[1];
        return rootMain(size, filename);
    }
    else {
        return slaveMain(rank, size);
    }
}
