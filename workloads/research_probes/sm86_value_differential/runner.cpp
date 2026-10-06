#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>

static void check(CUresult rc) {
    if (rc != CUDA_SUCCESS) { const char *s=nullptr; cuGetErrorString(rc,&s); fprintf(stderr,"CUDA driver: %s\n",s?s:"unknown"); std::exit(2); }
}
static uint32_t number(const char *s) {
    char *end=nullptr; errno=0; auto x=std::strtoull(s,&end,0);
    if(errno || !*s || *end || *s=='-' || x>UINT32_MAX) std::exit(2);
    return static_cast<uint32_t>(x);
}
int main(int argc,char **argv) {
    if(argc!=4) { fprintf(stderr,"usage: runner CUBIN DELTA SEED\n"); return 2; }
    uint32_t delta=number(argv[2]), state=number(argv[3]);
    uint32_t input[32], output[32];
    for(unsigned i=0;i<32;++i) { state=1664525u*state+1013904223u; input[i]=state; }
    const uint32_t edges[]={0,1,0x7fffffffu,0x80000000u,0xfffffffeu,0xffffffffu,delta-1,delta,delta+1};
    for(unsigned i=0;i<9;++i) input[i]=edges[i];
    check(cuInit(0)); CUdevice dev; check(cuDeviceGet(&dev,0));
    char name[256], bus[32]; CUuuid uuid; int major,minor,driver;
    check(cuDeviceGetName(name,256,dev)); check(cuDeviceGetPCIBusId(bus,32,dev));
    check(cuDeviceGetUuid(&uuid,dev)); check(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev));
    check(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev)); check(cuDriverGetVersion(&driver));
    if(major!=8 || minor!=6) { fprintf(stderr,"SM86 required\n"); return 2; }
    CUcontext ctx; check(cuDevicePrimaryCtxRetain(&ctx,dev)); check(cuCtxSetCurrent(ctx));
    CUmodule module; CUfunction kernel; check(cuModuleLoad(&module,argv[1])); check(cuModuleGetFunction(&kernel,module,"value_kernel"));
    CUdeviceptr in,out; check(cuMemAlloc(&in,sizeof(input))); check(cuMemAlloc(&out,sizeof(output)));
    check(cuMemcpyHtoD(in,input,sizeof(input))); check(cuMemsetD8(out,0xa5,sizeof(output)));
    void *args[]={&in,&out,&delta}; check(cuLaunchKernel(kernel,1,1,1,32,1,1,0,nullptr,args,nullptr));
    check(cuCtxSynchronize()); check(cuMemcpyDtoH(output,out,sizeof(output)));
    bool correct = true;
    for(unsigned i=0;i<32;++i) {
        const uint32_t expected = input[i] < delta ? input[i] + delta : input[i] - delta;
        if(output[i] != expected) correct = false;
    }
    printf("RESULT {\"device_index\":0,\"name\":\"%s\",\"pci\":\"%s\",\"cc\":\"%d.%d\",\"driver_api\":%d,\"uuid_hex\":\"",name,bus,major,minor,driver);
    for(auto b:uuid.bytes) printf("%02x",static_cast<unsigned char>(b));
    printf("\",\"input_address\":%llu,\"output_address\":%llu,\"delta\":%u,\"inputs\":[",(unsigned long long)in,(unsigned long long)out,delta);
    for(unsigned i=0;i<32;++i) printf("%s%u",i?",":"",input[i]);
    printf("],\"outputs\":["); for(unsigned i=0;i<32;++i) printf("%s%u",i?",":"",output[i]); printf("],\"correct\":%s}\n", correct ? "true" : "false");
    check(cuMemFree(in)); check(cuMemFree(out)); check(cuModuleUnload(module)); check(cuDevicePrimaryCtxRelease(dev));
    return correct ? 0 : 1;
}
