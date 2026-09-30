// Compile arbitrary NVIDIA PTX through the installed NVIDIA PTX Compiler API.
// This is a native PTX->cubin adapter, not an independent SASS compiler or ABI.

#include <nvPTXCompiler.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void check(nvPTXCompileResult result, const char* operation) {
  if (result != NVPTXCOMPILE_SUCCESS) {
    fail(std::string(operation) + " failed with nvPTXCompileResult=" +
         std::to_string(static_cast<int>(result)));
  }
}

std::string read_text(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open PTX input: " + path);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void write_binary(const std::string& path, const std::vector<char>& image) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) fail("cannot open cubin output: " + path);
  output.write(image.data(), static_cast<std::streamsize>(image.size()));
  if (!output) fail("cannot write cubin output: " + path);
}

std::string error_log(nvPTXCompilerHandle compiler) {
  size_t size = 0;
  if (nvPTXCompilerGetErrorLogSize(compiler, &size) != NVPTXCOMPILE_SUCCESS ||
      size == 0) {
    return {};
  }
  std::string log(size, '\0');
  if (nvPTXCompilerGetErrorLog(compiler, log.data()) != NVPTXCOMPILE_SUCCESS) {
    return {};
  }
  return log;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: nvptx_compile_sm86 INPUT.ptx OUTPUT.cubin [sm_86]\n";
    return 2;
  }
  const std::string input_path = argv[1];
  const std::string output_path = argv[2];
  const std::string arch = argc == 4 ? argv[3] : "sm_86";

  nvPTXCompilerHandle compiler = nullptr;
  try {
    const std::string ptx = read_text(input_path);
    check(nvPTXCompilerCreate(&compiler, ptx.size(), ptx.data()),
          "nvPTXCompilerCreate");
    const std::string option = "--gpu-name=" + arch;
    const char* options[] = {option.c_str()};
    const nvPTXCompileResult compile =
        nvPTXCompilerCompile(compiler, 1, options);
    if (compile != NVPTXCOMPILE_SUCCESS) {
      const std::string log = error_log(compiler);
      std::cerr << log;
      fail("nvPTXCompilerCompile failed with nvPTXCompileResult=" +
           std::to_string(static_cast<int>(compile)));
    }
    size_t image_size = 0;
    check(nvPTXCompilerGetCompiledProgramSize(compiler, &image_size),
          "nvPTXCompilerGetCompiledProgramSize");
    std::vector<char> image(image_size);
    check(nvPTXCompilerGetCompiledProgram(compiler, image.data()),
          "nvPTXCompilerGetCompiledProgram");
    write_binary(output_path, image);
    check(nvPTXCompilerDestroy(&compiler), "nvPTXCompilerDestroy");
    std::cout << "compiled " << input_path << " -> " << output_path
              << " (" << image.size() << " bytes, " << arch << ")\n";
    return 0;
  } catch (const std::exception& error) {
    if (compiler != nullptr) nvPTXCompilerDestroy(&compiler);
    std::cerr << "nvptx_compile_sm86: " << error.what() << '\n';
    return 1;
  }
}
