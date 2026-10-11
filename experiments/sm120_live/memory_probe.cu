// Timing and passive payload observations only. One measured launch per process.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#define CUDA(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); return 2;} } while(0)
struct Observation { unsigned long long cycles; unsigned value; };
template<int Mode> __global__ void memory_probe(unsigned *data, unsigned *dest, Observation *obs,
 unsigned words, unsigned rounds, unsigned stride, unsigned offset) {
 unsigned t=threadIdx.x+blockIdx.x*blockDim.x, v=0;
 unsigned long long begin=clock64();
 #pragma unroll 1
 for(unsigned r=0;r<rounds;r++) {
  unsigned ix=(t*stride+offset+r*blockDim.x*gridDim.x*stride)%words;
  if constexpr(Mode==0) { unsigned x; asm volatile("ld.global.ca.u32 %0,[%1];":"=r"(x):"l"(data+ix):"memory");v+=x; }
  if constexpr(Mode==1) { unsigned x; asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(x):"l"(data+ix):"memory");v+=x; }
  if constexpr(Mode==2) { unsigned x;asm volatile("ld.global.ca.u32 %0,[%1];":"=r"(x):"l"(data+ix):"memory");v+=x;asm volatile("st.global.u32 [%0],%1;"::"l"(dest+ix),"r"(x+r):"memory"); }
  if constexpr(Mode==3) { asm volatile("st.global.u32 [%0],%1;"::"l"(dest+ix),"r"(r+t):"memory"); }
  if constexpr(Mode==4) { asm volatile("st.global.u8 [%0],%1;"::"l"((unsigned char*)dest+ix*4),"r"(r+t):"memory"); }
  if constexpr(Mode==5) { unsigned x;asm volatile("atom.global.add.u32 %0,[%1],1;":"=r"(x):"l"(data+(stride? t%words:0)):"memory");v+=x; }
  if constexpr(Mode==6) { asm volatile("red.global.add.u32 [%0],1;"::"l"(data+(stride? t%words:0)):"memory"); }
  if constexpr(Mode==10) { asm volatile("red.global.add.u32 [%0],1;"::"l"(data+(stride? t%words:0)):"memory"); asm volatile("membar.gl;":::"memory"); unsigned x; asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(x):"l"(data+(stride? t%words:0)):"memory"); v+=x; }
 }
 // Sink dependency orders the stop timer after load/returning-atomic result.
 unsigned long long end;
 asm volatile("{.reg .pred p;setp.ne.u32 p,%1,0;@p mov.u64 %0,%%clock64;@!p mov.u64 %0,%%clock64;}" : "=l"(end):"r"(v):"memory");
 obs[t]={end-begin,v};
}
template<int Groups> __global__ void async_probe(const unsigned *src, unsigned *out, Observation *obs, unsigned rounds) {
 __shared__ __align__(128) unsigned tiles[Groups][32*4];
 unsigned t=threadIdx.x;
 unsigned smem=(unsigned)__cvta_generic_to_shared(&tiles[0][t*4]);
 unsigned sum=0; unsigned long long begin=clock64();
 #pragma unroll 1
 for(unsigned r=0;r<rounds;r++) {
  #pragma unroll
  for(int g=0;g<Groups;g++) {
   unsigned dst=smem+g*32*16;
   const unsigned *p=src+g*128+t*4;
   asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(p):"memory");
   asm volatile("cp.async.commit_group;":::"memory");
  }
  asm volatile("cp.async.wait_group 0;":::"memory");
  #pragma unroll
  for(int g=0;g<Groups;g++) sum+=tiles[g][t*4];
 }
 out[t]=sum;obs[t]={clock64()-begin,sum};
}
int main(int argc,char**argv){
 if(argc!=8){fprintf(stderr,"usage: probe mode words rounds stride offset threads blocks\n");return 2;}
 int mode=atoi(argv[1]);unsigned words=strtoul(argv[2],0,0),rounds=atoi(argv[3]),stride=atoi(argv[4]),offset=atoi(argv[5]),threads=atoi(argv[6]),blocks=atoi(argv[7]);
 if(!words||!rounds||!threads||threads>1024||!blocks||mode<0||mode>10)return 2;
 cudaDeviceProp prop;CUDA(cudaGetDeviceProperties(&prop,0));
 unsigned *data,*dest;Observation*obs;size_t count=(size_t)threads*blocks;
 words=std::max(words,512u);std::vector<unsigned> h(words);for(unsigned i=0;i<words;i++)h[i]=i*37+11;
 CUDA(cudaMalloc(&data,words*4));CUDA(cudaMalloc(&dest,words*4));CUDA(cudaMalloc(&obs,count*sizeof(Observation)));
 CUDA(cudaMemcpy(data,h.data(),words*4,cudaMemcpyHostToDevice));CUDA(cudaMemset(dest,0,words*4));
 // Resolve/load the selected kernel before the event bracket. Lazy module
 // loading after the start event otherwise adds host submission idle time.
 cudaFuncAttributes attributes;
 switch(mode){
#define PRELOAD(N) case N: CUDA(cudaFuncGetAttributes(&attributes,memory_probe<N>));break;
 PRELOAD(0) PRELOAD(1) PRELOAD(2) PRELOAD(3) PRELOAD(4) PRELOAD(5) PRELOAD(6)
#undef PRELOAD
 case 7:CUDA(cudaFuncGetAttributes(&attributes,async_probe<1>));break;
 case 8:CUDA(cudaFuncGetAttributes(&attributes,async_probe<2>));break;
 case 9:CUDA(cudaFuncGetAttributes(&attributes,async_probe<4>));break;
 }
 cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));CUDA(cudaDeviceSynchronize());
 CUDA(cudaProfilerStart());CUDA(cudaEventRecord(a));
 switch(mode){
#define M(N) case N:memory_probe<N><<<blocks,threads>>>(data,dest,obs,words,rounds,stride,offset);break;
 M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(10)
#undef M
 case 7:async_probe<1><<<1,32>>>(data,dest,obs,rounds);break;
 case 8:async_probe<2><<<1,32>>>(data,dest,obs,rounds);break;
 case 9:async_probe<4><<<1,32>>>(data,dest,obs,rounds);break;
 }
 CUDA(cudaGetLastError());CUDA(cudaEventRecord(b));CUDA(cudaEventSynchronize(b));CUDA(cudaProfilerStop());float ms;CUDA(cudaEventElapsedTime(&ms,a,b));
 std::vector<Observation> result(count);CUDA(cudaMemcpy(result.data(),obs,count*sizeof(Observation),cudaMemcpyDeviceToHost));
 std::vector<unsigned long long> cycles;for(auto x:result)cycles.push_back(x.cycles);std::sort(cycles.begin(),cycles.end());
 unsigned sample=0;CUDA(cudaMemcpy(&sample,dest,4,cudaMemcpyDeviceToHost));
 int cluster=0;CUDA(cudaDeviceGetAttribute(&cluster,cudaDevAttrClusterLaunch,0));
 if(mode>=7) printf("SM120_PAYLOAD_OBSERVED mode=%d value=%u after_event_sync=1\n",mode,sample);
 printf("{\"mode\":%d,\"words\":%u,\"rounds\":%u,\"stride\":%u,\"offset\":%u,\"threads\":%u,\"blocks\":%u,\"event_ms\":%.9f,\"clock_cycles_median\":%llu,\"payload_sample\":%u,\"gpu\":\"%s\",\"cc\":%d,\"cluster_launch\":%d}\n",mode,words,rounds,stride,offset,threads,blocks,ms,cycles[cycles.size()/2],sample,prop.name,prop.major*10+prop.minor,cluster);
 CUDA(cudaFree(data));CUDA(cudaFree(dest));CUDA(cudaFree(obs));return 0;
}
