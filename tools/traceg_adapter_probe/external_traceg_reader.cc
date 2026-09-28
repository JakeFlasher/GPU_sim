// Independent legacy .traceg reader for the bounded SM86 adapter probe.
//
// This file intentionally uses the pinned Accel-Sim/GPGPU-Sim trace-parser
// API, rather than the Python parser in adapter.py.  It is a structural
// representation check only: it does not model timing or validate silicon.

#include <cuda_runtime.h>
#include "trace_parser.h"

#include <cstdint>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Canonical {
  unsigned pc = 0;
  unsigned mask = 0;
  std::string opcode;
  std::vector<unsigned> dest;
  std::vector<unsigned> src;
  int64_t immediate = 0;
};

struct LaunchHeader {
  std::string kernel_name;
  unsigned kernel_id = 0;
  unsigned shared_memory_bytes = 0;
  unsigned registers_per_thread = 0;
  unsigned binary_version = 0;
  unsigned stream_id = 0;
  uint64_t shared_base_address = 0;
  uint64_t local_base_address = 0;
  std::string nvbit_version;
  unsigned accelsim_tracer_version = 0;
  unsigned lineinfo = 0;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

bool starts_with(const std::string &value, const char *prefix) {
  return value.rfind(prefix, 0) == 0;
}

template <typename T>
T parse_unsigned_field(const std::string &value, int base, const char *label) {
  size_t consumed = 0;
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value, &consumed, base);
  } catch (const std::exception &) {
    fail(std::string("malformed ") + label);
  }
  if (consumed != value.size() || parsed > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
    fail(std::string("malformed ") + label);
  }
  return static_cast<T>(parsed);
}

std::string header_value(const std::string &line, const char *prefix) {
  if (!starts_with(line, prefix)) fail("malformed header prefix");
  return line.substr(std::char_traits<char>::length(prefix));
}

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t");
  return value.substr(first, last - first + 1);
}

std::string json_string(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    switch (c) {
    case '"': out << "\\\""; break;
    case '\\': out << "\\\\"; break;
    case '\n': out << "\\n"; break;
    case '\r': out << "\\r"; break;
    case '\t': out << "\\t"; break;
    default:
      if (c < 0x20) {
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<unsigned>(c) << std::dec << std::setfill(' ');
      } else {
        out << static_cast<char>(c);
      }
    }
  }
  out << '"';
  return out.str();
}

void emit_registers(std::ostringstream &out, const std::vector<unsigned> &regs) {
  out << '[';
  for (size_t i = 0; i < regs.size(); ++i) {
    if (i) out << ',';
    out << '"' << 'R' << regs[i] << '"';
  }
  out << ']';
}

void emit_canonical(std::ostringstream &out, const Canonical &record) {
  out << "{\"block\":[0,0,0],\"warp\":0,\"pc\"" << ":" << record.pc
      << ",\"mask\":" << record.mask << ",\"opcode\":"
      << json_string(record.opcode) << ",\"dest_regs\":";
  emit_registers(out, record.dest);
  out << ",\"src_regs\":";
  emit_registers(out, record.src);
  out << ",\"immediate\":" << record.immediate << ",\"mem_width\":0,"
      << "\"addresses\":[]}";
}

Canonical parse_record(const std::string &line) {
  inst_trace_t record;
  if (!record.parse_from_string(line, 5, 0, dim3(0, 0, 0), dim3(0, 0, 0),
                               dim3(0, 0, 0), 0)) {
    fail("Accel-Sim parser returned false");
  }
  if (record.reg_dsts_num > MAX_DST || record.reg_srcs_num > MAX_SRC) {
    fail("legacy parser record exceeds register arity");
  }
  // The upstream parser is intentionally permissive about fields introduced
  // by later trace versions.  This probe is not: accepting trailing tokens
  // would allow a malformed trace to compare equal after the parser silently
  // ignored the extra data.  Validate the exact v5 ALU grammar before using
  // any parsed values.
  std::vector<std::string> fields;
  {
    std::istringstream tokens(line);
    std::string token;
    while (tokens >> token) fields.push_back(token);
  }
  const size_t expected = 3u + record.reg_dsts_num + 1u + 1u +
                          record.reg_srcs_num + 2u;
  if (fields.size() != expected) {
    fail("trace record has trailing or missing fields");
  }
  Canonical result;
  result.pc = record.m_pc;
  result.mask = record.mask;
  result.opcode = record.opcode;
  if (result.opcode != "IADD3" && result.opcode != "FADD" && result.opcode != "EXIT") {
    // The legacy parser stores arbitrary opcode text and defers architecture
    // lookup to trace-driven execution.  The independent representation
    // probe must fail closed before that later stage can turn an unknown
    // token into a default/similar operation.
    fail("opcode is outside the admitted SM86 common subset");
  }
  // The legacy struct stores `imm` as uint64_t, while the adapter contract
  // admits signed 64-bit IADD3 literals.  Recover the signed spelling from
  // the externally parsed line rather than allowing -8 to become
  // UINT64_MAX-7 in the canonical result.
  std::string immediate_text;
  {
    if (fields.empty()) {
      fail("missing immediate field");
    }
    immediate_text = fields.back();
  }
  try {
    size_t consumed = 0;
    result.immediate = std::stoll(immediate_text, &consumed, 10);
    if (consumed != immediate_text.size()) fail("immediate has trailing data");
  } catch (const std::exception &) {
    fail("immediate is not signed decimal");
  }
  for (unsigned i = 0; i < record.reg_dsts_num; ++i) {
    if (record.reg_dest[i].type != REG) fail("non-regular destination register");
    result.dest.push_back(record.reg_dest[i].num);
  }
  for (unsigned i = 0; i < record.reg_srcs_num; ++i) {
    if (record.reg_src[i].type != REG) fail("non-regular source register");
    result.src.push_back(record.reg_src[i].num);
  }
  if (record.memadd_info || record.tma_memadd_info) {
    fail("memory record is outside the ALU-only external probe");
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) fail("usage: external_traceg_reader TRACEG");
    std::ifstream input(argv[1]);
    if (!input) fail("cannot open traceg");

    std::string line;
    bool in_tb = false;
    bool saw_begin = false;
    bool saw_end = false;
    bool saw_exit = false;
    unsigned declared = 0;
    unsigned block_fields = 0;
    unsigned warp_fields = 0;
    unsigned binary_version = 0;
    bool saw_insts = false;
    bool saw_cluster_id = false;
    bool saw_cluster_cta = false;
    bool saw_cluster_rank = false;
    LaunchHeader header;
    bool saw_kernel_name = false;
    bool saw_kernel_id = false;
    bool saw_grid = false;
    bool saw_block = false;
    bool saw_shmem = false;
    bool saw_nregs = false;
    bool saw_binary = false;
    bool saw_stream = false;
    bool saw_shared_base = false;
    bool saw_local_base = false;
    bool saw_nvbit = false;
    bool saw_accelsim = false;
    bool saw_lineinfo = false;
    std::vector<Canonical> records;
    std::optional<Canonical> terminal;

    while (std::getline(input, line)) {
      if (line == "#BEGIN_TB") {
        if (in_tb || saw_begin) fail("duplicate/nested TB marker");
        in_tb = true;
        saw_begin = true;
        continue;
      }
      if (line == "#END_TB") {
        if (!in_tb || saw_end) fail("malformed TB end marker");
        in_tb = false;
        saw_end = true;
        continue;
      }
      if (line.empty() || line[0] == '#') continue;
      if (starts_with(line, "-kernel name =")) {
        if (in_tb || saw_kernel_name) fail("duplicate/misplaced kernel name header");
        header.kernel_name = trim(header_value(line, "-kernel name ="));
        if (header.kernel_name.empty()) fail("empty kernel name header");
        saw_kernel_name = true;
        continue;
      }
      if (starts_with(line, "-kernel id =")) {
        if (in_tb || saw_kernel_id) fail("duplicate/misplaced kernel id header");
        header.kernel_id = parse_unsigned_field<unsigned>(trim(header_value(line, "-kernel id =")), 10, "kernel id");
        saw_kernel_id = true;
        continue;
      }
      if (starts_with(line, "-grid dim =")) {
        if (in_tb || saw_grid || line != "-grid dim = (1,1,1)") fail("unsupported grid header");
        saw_grid = true;
        continue;
      }
      if (starts_with(line, "-block dim =")) {
        if (in_tb || saw_block || line != "-block dim = (32,1,1)") fail("unsupported block header");
        saw_block = true;
        continue;
      }
      if (starts_with(line, "-shmem =")) {
        if (in_tb || saw_shmem) fail("duplicate/misplaced shared-memory header");
        header.shared_memory_bytes = parse_unsigned_field<unsigned>(trim(header_value(line, "-shmem =")), 10, "shared memory");
        saw_shmem = true;
        continue;
      }
      if (starts_with(line, "-nregs =")) {
        if (in_tb || saw_nregs) fail("duplicate/misplaced register header");
        header.registers_per_thread = parse_unsigned_field<unsigned>(trim(header_value(line, "-nregs =")), 10, "register count");
        saw_nregs = true;
        continue;
      }
      if (starts_with(line, "-binary version =")) {
        if (in_tb || saw_binary) fail("duplicate/misplaced binary header");
        header.binary_version = parse_unsigned_field<unsigned>(trim(header_value(line, "-binary version =")), 10, "binary version");
        binary_version = header.binary_version;
        if (binary_version != 86) fail("traceg is not SM86");
        saw_binary = true;
        continue;
      }
      if (starts_with(line, "-cuda stream id =")) {
        if (in_tb || saw_stream) fail("duplicate/misplaced stream header");
        header.stream_id = parse_unsigned_field<unsigned>(trim(header_value(line, "-cuda stream id =")), 10, "stream id");
        saw_stream = true;
        continue;
      }
      if (starts_with(line, "-shmem base_addr =")) {
        if (in_tb || saw_shared_base) fail("duplicate/misplaced shared base header");
        std::string value = trim(header_value(line, "-shmem base_addr ="));
        if (value.rfind("0x", 0) != 0 && value.rfind("0X", 0) != 0) fail("shared base is not hexadecimal");
        header.shared_base_address = parse_unsigned_field<uint64_t>(value, 0, "shared base");
        saw_shared_base = true;
        continue;
      }
      if (starts_with(line, "-local mem base_addr =")) {
        if (in_tb || saw_local_base) fail("duplicate/misplaced local base header");
        std::string value = trim(header_value(line, "-local mem base_addr ="));
        if (value.rfind("0x", 0) != 0 && value.rfind("0X", 0) != 0) fail("local base is not hexadecimal");
        header.local_base_address = parse_unsigned_field<uint64_t>(value, 0, "local base");
        saw_local_base = true;
        continue;
      }
      if (starts_with(line, "-nvbit version =")) {
        if (in_tb || saw_nvbit) fail("duplicate/misplaced NVBit header");
        header.nvbit_version = trim(header_value(line, "-nvbit version ="));
        if (header.nvbit_version.empty()) fail("empty NVBit version");
        saw_nvbit = true;
        continue;
      }
      if (starts_with(line, "-accelsim tracer version =")) {
        if (in_tb || saw_accelsim) fail("duplicate/misplaced Accel-Sim version header");
        header.accelsim_tracer_version = parse_unsigned_field<unsigned>(trim(header_value(line, "-accelsim tracer version =")), 10, "Accel-Sim version");
        saw_accelsim = true;
        continue;
      }
      if (starts_with(line, "-enable lineinfo =")) {
        if (in_tb || saw_lineinfo) fail("duplicate/misplaced lineinfo header");
        header.lineinfo = parse_unsigned_field<unsigned>(trim(header_value(line, "-enable lineinfo =")), 10, "lineinfo");
        if (header.lineinfo != 0) fail("lineinfo is outside the probe contract");
        saw_lineinfo = true;
        continue;
      }
      if (starts_with(line, "insts =")) {
        if (!in_tb || saw_insts) fail("duplicate/misplaced insts declaration");
        std::stringstream ss(line.substr(line.find('=') + 1));
        ss >> declared;
        if (!ss || declared == 0) fail("invalid insts declaration");
        ss >> std::ws;
        if (!ss.eof()) fail("insts declaration has trailing fields");
        saw_insts = true;
        continue;
      }
      if (starts_with(line, "thread block =")) {
        if (line != "thread block = 0,0,0") fail("unsupported thread block");
        ++block_fields;
        continue;
      }
      if (starts_with(line, "warp =")) {
        if (line != "warp = 0") fail("unsupported warp");
        ++warp_fields;
        continue;
      }
      if (starts_with(line, "cluster id =")) {
        if (!in_tb || saw_cluster_id || line != "cluster id = 0,0,0") fail("malformed cluster id");
        saw_cluster_id = true;
        continue;
      }
      if (starts_with(line, "cluster cta =")) {
        if (!in_tb || saw_cluster_cta || line != "cluster cta = 0,0,0") fail("malformed cluster cta");
        saw_cluster_cta = true;
        continue;
      }
      if (starts_with(line, "cluster rank =")) {
        if (!in_tb || saw_cluster_rank || line != "cluster rank = 0") fail("malformed cluster rank");
        saw_cluster_rank = true;
        continue;
      }
      if (line[0] == '-') fail("unknown or unvalidated traceg header");
      if (!in_tb) fail("record outside TB");

      std::istringstream tokens(line);
      std::string pc_text, mask_text, dst_count, first_opcode;
      tokens >> pc_text >> mask_text >> dst_count >> first_opcode;
      if (!tokens) fail("malformed trace record");
      if (first_opcode == "EXIT") {
        Canonical parsed_exit = parse_record(line);
        if (parsed_exit.opcode != "EXIT") fail("legacy parser did not retain EXIT opcode");
        std::string rest;
        std::getline(tokens, rest);
        unsigned exit_pc = 0;
        std::stringstream pc_stream(pc_text);
        pc_stream >> std::hex >> exit_pc;
        if (!pc_stream) fail("malformed EXIT PC");
        unsigned mask = 1;
        std::stringstream mask_stream(mask_text);
        mask_stream >> std::hex >> mask;
        if (!mask_stream || mask != 0 || dst_count != "0") fail("malformed EXIT");
        unsigned src_count = 1;
        std::istringstream exit_fields(rest);
        exit_fields >> src_count;
        if (!exit_fields || src_count != 0) fail("malformed EXIT source count");
        std::string width, immediate;
        exit_fields >> width >> immediate;
        if (!exit_fields || width != "0" || immediate != "0") fail("malformed EXIT fields");
        if (saw_exit) fail("duplicate EXIT");
        saw_exit = true;
        terminal = parsed_exit;
        continue;
      }
      if (saw_exit) fail("record follows EXIT");
      records.push_back(parse_record(line));
    }

    if (in_tb || !saw_begin || !saw_end || !saw_exit || declared != records.size() + 1 ||
        block_fields != 1 || warp_fields != 1 || binary_version != 86 || records.empty() ||
        !saw_insts || !saw_cluster_id || !saw_cluster_cta || !saw_cluster_rank ||
        !saw_kernel_name || !saw_kernel_id || !saw_grid || !saw_block || !saw_shmem ||
        !saw_nregs || !saw_binary || !saw_stream || !saw_shared_base || !saw_local_base ||
        !saw_nvbit || !saw_accelsim || !saw_lineinfo) {
      fail("traceg structural contract failed");
    }

    std::ostringstream out;
    out << "{\"schema\":\"accel_sim_traceg_reader_v1\",\"status\":\"ok\","
        << "\"termination\":\"explicit_exit\",\"record_count\":" << records.size()
        << ",\"launch\":{";
    out << "\"kernel_name\":" << json_string(header.kernel_name)
        << ",\"kernel_id\":" << header.kernel_id
        << ",\"grid\":[1,1,1],\"block\":[32,1,1]"
        << ",\"shared_memory_bytes\":" << header.shared_memory_bytes
        << ",\"registers_per_thread\":" << header.registers_per_thread
        << ",\"binary_version\":" << header.binary_version
        << ",\"stream_id\":" << header.stream_id
        << ",\"shared_base_address\":" << header.shared_base_address
        << ",\"local_base_address\":" << header.local_base_address
        << ",\"nvbit_version\":" << json_string(header.nvbit_version)
        << ",\"accelsim_tracer_version\":" << header.accelsim_tracer_version
        << ",\"lineinfo\":" << header.lineinfo << "},\"records\":[";
    for (size_t i = 0; i < records.size(); ++i) {
      if (i) out << ',';
      emit_canonical(out, records[i]);
    }
    out << "],\"terminal\":";
    if (terminal.has_value()) emit_canonical(out, *terminal);
    else out << "null";
    out << "}\n";
    std::cout << out.str();
    std::cerr << "records=" << records.size() << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "external_traceg_reader_refused: " << error.what() << "\n";
    return 2;
  }
}
