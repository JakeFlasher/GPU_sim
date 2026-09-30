// Link a PTX text input into an SM86 cubin with NVIDIA nvJitLink.
// This is an official NVIDIA PTX/linker path, not an independent SASS or ABI
// implementation.

#include <nvJitLink.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void check(nvJitLinkResult result, const char* operation,
           nvJitLinkHandle handle = nullptr) {
  if (result == NVJITLINK_SUCCESS) return;
  std::string message = std::string(operation) +
                        " failed with nvJitLinkResult=" +
                        std::to_string(static_cast<int>(result));
  if (handle != nullptr) {
    size_t size = 0;
    if (nvJitLinkGetErrorLogSize(handle, &size) == NVJITLINK_SUCCESS &&
        size != 0) {
      std::string log(size, '\0');
      if (nvJitLinkGetErrorLog(handle, log.data()) == NVJITLINK_SUCCESS) {
        message += ":\n" + log;
      }
    }
  }
  fail(message);
}

std::vector<char> read_binary(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open input: " + path);
  return std::vector<char>(std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>());
}

void write_binary(const std::string& path, const void* data, size_t size) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) fail("cannot open output: " + path);
  output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!output) fail("cannot write output: " + path);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: nvjitlink_sm86 INPUT.ptx OUTPUT.cubin [sm_86]\n";
    return 2;
  }
  const std::string input_path = argv[1];
  const std::string output_path = argv[2];
  const std::string arch = argc == 4 ? argv[3] : "sm_86";
  nvJitLinkHandle linker = nullptr;
  try {
    std::vector<char> input = read_binary(input_path);
    // Unlike length-delimited nvPTXCompiler input, the linker consumes PTX
    // text with a terminating NUL. Without it, valid PTX may be rejected.
    input.push_back('\0');
    const std::string arch_option = "-arch=" + arch;
    const char* options[] = {arch_option.c_str()};
    check(nvJitLinkCreate(&linker, 1, options), "nvJitLinkCreate");
    check(nvJitLinkAddData(linker, NVJITLINK_INPUT_PTX, input.data(),
                           input.size(), input_path.c_str()),
          "nvJitLinkAddData", linker);
    check(nvJitLinkComplete(linker), "nvJitLinkComplete", linker);
    size_t image_size = 0;
    check(nvJitLinkGetLinkedCubinSize(linker, &image_size),
          "nvJitLinkGetLinkedCubinSize", linker);
    std::vector<char> image(image_size);
    check(nvJitLinkGetLinkedCubin(linker, image.data()),
          "nvJitLinkGetLinkedCubin", linker);
    write_binary(output_path, image.data(), image.size());
    check(nvJitLinkDestroy(&linker), "nvJitLinkDestroy");
    std::cout << "linked " << input_path << " -> " << output_path << " ("
              << image.size() << " bytes, " << arch << ")\n";
    return 0;
  } catch (const std::exception& error) {
    if (linker != nullptr) nvJitLinkDestroy(&linker);
    std::cerr << "nvjitlink_sm86: " << error.what() << '\n';
    return 1;
  }
}
