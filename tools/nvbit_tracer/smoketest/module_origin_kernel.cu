// Both binaries export the same name but have different behavior. The smoke
// must identify the loaded owner, not choose a file by kernel name.
#ifndef ORIGIN_VALUE
#error ORIGIN_VALUE must be defined
#endif
extern "C" __global__ void origin_collision(unsigned* output) {
  output[threadIdx.x] = ORIGIN_VALUE + threadIdx.x;
}
