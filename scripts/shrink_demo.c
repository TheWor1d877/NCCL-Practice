/* ncclCommShrink 最小示例
 * 2 rank 建父通信器 -> rank1 模拟故障退出 -> rank0 缩容后继续 allreduce
 * 编译: mpicc shrink_demo.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lnccl -o shrink_demo
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CK(call) do { \
  int _r = (int)(call); \
  if (_r != 0) { printf("FAIL %s:%d rc=%d\n", __FILE__, __LINE__, _r); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, 1); } \
} while (0)

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  CK(cudaSetDevice(rank % 2));

  size_t count = 1 << 20;                 /* 1M float = 4MB */
  size_t bytes = count * sizeof(float);
  float *sendbuf = NULL, *recvbuf = NULL;
  CK(cudaMalloc(&sendbuf, bytes));
  CK(cudaMalloc(&recvbuf, bytes));
  float *h = (float *)malloc(bytes);
  for (size_t i = 0; i < count; i++) h[i] = (float)(rank + 1);
  CK(cudaMemcpy(sendbuf, h, bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CK(cudaStreamCreate(&stream));

  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t parent = NULL;
  ncclResult_t r = ncclCommInitRank(&parent, nranks, id, rank);
  printf("rank %d: parent comm init rc=%d nranks=%d\n", rank, (int)r, nranks);
  fflush(stdout);

  r = ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, parent, stream);
  CK(cudaStreamSynchronize(stream));
  float val = 0.0f;
  CK(cudaMemcpy(&val, recvbuf, sizeof(float), cudaMemcpyDeviceToHost));
  printf("rank %d: parent allreduce rc=%d result[0]=%.1f (expect %.1f)\n",
         rank, (int)r, val, (float)(nranks * (nranks + 1) / 2));
  fflush(stdout);

  if (rank != 0) {
    printf("rank %d: simulating failure, exiting without destroying parent comm\n", rank);
    fflush(stdout);
    MPI_Finalize();
    return 0;
  }

  int exclude[1] = {1};
  ncclComm_t shrunk = NULL;
  r = ncclCommShrink(parent, exclude, 1, &shrunk, NULL, NCCL_SHRINK_ABORT);
  printf("rank 0: ncclCommShrink rc=%d (%s)\n", (int)r, ncclGetErrorString(r));
  fflush(stdout);
  if (r != ncclSuccess) { MPI_Finalize(); return 1; }

  int srank = -1, snranks = -1;
  ncclCommUserRank(shrunk, &srank);
  ncclCommCount(shrunk, &snranks);
  printf("rank 0: shrunk comm size=%d myrank=%d\n", snranks, srank);

  r = ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, shrunk, stream);
  CK(cudaStreamSynchronize(stream));
  CK(cudaMemcpy(&val, recvbuf, sizeof(float), cudaMemcpyDeviceToHost));
  printf("rank 0: shrunk allreduce rc=%d result[0]=%.1f (only remaining rank, expect 1.0)\n",
         (int)r, val);
  printf("rank 0: SUCCESS - shrink done without re-generating uniqueId\n");
  fflush(stdout);

  ncclCommDestroy(shrunk);
  MPI_Finalize();
  return 0;
}
