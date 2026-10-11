#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
namespace cg=cooperative_groups;
#define CUDA(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 2;}}while(0)
__global__ __cluster_dims__(2,1,1) void cluster_probe(unsigned *out, unsigned *ticks) {
  auto c=cg::this_cluster();
  unsigned rank=c.block_rank();
  unsigned long long a=clock64();
  if(threadIdx.x==0) out[blockIdx.x]=rank;
  c.sync();
  unsigned long long b=clock64();
  if(threadIdx.x==0) ticks[blockIdx.x]=(unsigned)(b-a);
}
int main(){
 cudaDeviceProp p;CUDA(cudaGetDeviceProperties(&p,0));
 unsigned *out,*ticks;CUDA(cudaMalloc(&out,2*sizeof(unsigned)));CUDA(cudaMalloc(&ticks,2*sizeof(unsigned)));CUDA(cudaMemset(out,0,2*sizeof(unsigned)));CUDA(cudaMemset(ticks,0,2*sizeof(unsigned)));
 cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));CUDA(cudaEventRecord(a));cluster_probe<<<2,32>>>(out,ticks);CUDA(cudaGetLastError());CUDA(cudaEventRecord(b));CUDA(cudaEventSynchronize(b));float ms;CUDA(cudaEventElapsedTime(&ms,a,b));
 unsigned h[2],t[2];CUDA(cudaMemcpy(h,out,8,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(t,ticks,8,cudaMemcpyDeviceToHost));
 printf("SM120_CLUSTER_PAYLOAD rank0=%u rank1=%u ticks0=%u ticks1=%u\n",h[0],h[1],t[0],t[1]);
 printf("{\"event_ms\":%.9f,\"rank0\":%u,\"rank1\":%u,\"ticks0\":%u,\"ticks1\":%u,\"cluster_launch_attr\":%d,\"gpu\":\"%s\",\"cc\":%d}\n",ms,h[0],h[1],t[0],t[1],p.clusterLaunch,p.name,p.major*10+p.minor);
 CUDA(cudaFree(out));CUDA(cudaFree(ticks));return 0;
}
