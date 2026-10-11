// SPDX-License-Identifier: BSD-3-Clause
//
// Bounded static metadata writer for the SM120 NVBit producer.  This header
// deliberately consumes only the bytes and SASS text returned by NVBit's
// Instr API; it does not invoke cuobjdump/nvdisasm, infer pipeline latencies,
// or rewrite an instruction into an architecture-equivalent mnemonic.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace nvbit_static_metadata {

struct Instruction {
  uint32_t pc = 0;
  std::string sass;
  std::string opcode;
  std::string opcode_short;
  bool predicated = false;
  bool predicate_uniform = false;
  bool predicate_negated = false;
  int predicate_register = -1;
  bool predicate_register_available = false;
  std::vector<uint8_t> encoding;
  struct Operand {
    std::string text;
    std::string type;
    bool is_negative = false;
    bool is_not = false;
    bool is_absolute = false;
    bool reuse_bit = false;
    std::vector<std::string> modifiers;
    int nbytes = 0;
    bool has_u64 = false;
    uint64_t u64 = 0;
    bool has_f64 = false;
    double f64 = 0.0;
    bool has_reg = false;
    int reg_num = 0;
    bool legacy_reg_alias_ambiguous = false;
    bool has_reg_property = false;
    std::string reg_property;
    struct RegisterRef {
      std::string role;
      int number = 0;
      std::string modifier;
      bool has_modifier = false;
    };
    std::vector<RegisterRef> register_refs;
    bool has_pred = false;
    int pred_num = 0;
    bool has_cbank = false;
    int cbank_id = 0;
    bool cbank_has_imm = false;
    int cbank_imm = 0;
    bool cbank_has_reg = false;
    int cbank_reg = 0;
    bool has_mref = false;
    bool mref_has_ra = false;
    int mref_ra = 0;
    std::string mref_ra_modifier;
    bool mref_has_ur = false;
    int mref_ur = 0;
    std::string mref_ur_modifier;
    bool mref_has_imm = false;
    int mref_imm = 0;
    bool mref_has_desc = false;
    int mref_desc = 0;
    bool has_generic = false;
    std::string generic_text;
    bool has_desc = false;
    int desc_ureg = 0;
    bool has_tma = false;
    int tma_ureg[4] = {-1, -1, -1, -1};
  };
  std::vector<Operand> operands;
};

struct Function {
  uint32_t function_id = 0;
  std::string name;
  uint64_t function_address = 0;
  uint32_t architecture_version = 0;
  std::vector<Instruction> instructions;
};

inline std::string JsonEscape(const std::string &text) {
  std::ostringstream out;
  for (unsigned char c : text) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned int>(c) << std::dec;
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

inline std::string HexBytes(const std::vector<uint8_t> &bytes,
                            bool reverse) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  if (reverse) {
    for (auto it = bytes.rbegin(); it != bytes.rend(); ++it)
      out << std::setw(2) << static_cast<unsigned int>(*it);
  } else {
    for (uint8_t byte : bytes)
      out << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return out.str();
}

inline std::string HexWord(const std::vector<uint8_t> &bytes,
                           size_t begin, size_t count) {
  if (begin >= bytes.size()) return std::string();
  const size_t end = std::min(bytes.size(), begin + count);
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (size_t i = end; i > begin; --i)
    out << std::setw(2) << static_cast<unsigned int>(bytes[i - 1]);
  return out.str();
}

inline bool ParseHexU64(const std::string &hex, uint64_t *value) {
  if (value == nullptr || hex.empty() || hex.size() > 16) return false;
  uint64_t parsed = 0;
  for (char c : hex) {
    unsigned digit = 0;
    if (c >= '0' && c <= '9') digit = static_cast<unsigned>(c - '0');
    else if (c >= 'a' && c <= 'f') digit = static_cast<unsigned>(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') digit = static_cast<unsigned>(c - 'A' + 10);
    else return false;
    if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 16u)
      return false;
    parsed = parsed * 16u + digit;
  }
  *value = parsed;
  return true;
}

// Decode the legacy Volta+-style control fields from the raw high 64-bit word
// returned by NVBit's callback.  The Blackwell callback layout is verified by
// the retained native encoding records: bytes [8,15] are the word whose bits
// 41+ contain the control code.  The exporter retains both raw words and
// labels this decode as an observed bit extraction; it does not claim that
// these fields model any undocumented Blackwell scheduling mechanism.
inline void WriteControlFields(std::ostream &out,
                               const std::vector<uint8_t> &encoding,
                               uint32_t architecture_version) {
  if (architecture_version < 70 || encoding.size() < 16) {
    out << ",\"control_fields_available\":false"
        << ",\"control_fields_unavailable_reason\":\"encoding_short_or_pre_volta\"";
    return;
  }
  const std::string control_hex = HexWord(encoding, 8, 8);
  uint64_t control_word = 0;
  if (!ParseHexU64(control_hex, &control_word)) {
    out << ",\"control_fields_available\":false"
        << ",\"control_fields_unavailable_reason\":\"control_word_parse_failed\"";
    return;
  }
  const uint64_t control = control_word >> 41;
  out << ",\"control_fields_available\":true"
      << ",\"control_word_hex\":\"" << control_hex << "\""
      << ",\"control_word_shift\":41"
      << ",\"stall_count\":" << (control & 0xfu)
      << ",\"yield_bit\":" << ((control >> 4) & 1u)
      << ",\"new_write_barrier_id\":" << ((control >> 5) & 7u)
      << ",\"new_read_barrier_id\":" << ((control >> 8) & 7u)
      << ",\"wait_barrier_mask\":" << ((control >> 11) & 0x3fu)
      << ",\"control_fields\":{\"stall_count\":"
      << (control & 0xfu)
      << ",\"is_yield\":" << (((control >> 4) & 1u) == 0 ? "true" : "false")
      << ",\"is_new_read_barrier\":"
      << (((control >> 8) & 7u) != 7u ? "true" : "false")
      << ",\"is_new_write_barrier\":"
      << (((control >> 5) & 7u) != 7u ? "true" : "false")
      << ",\"id_new_read_barrier\":" << ((control >> 8) & 7u)
      << ",\"id_new_write_barrier\":" << ((control >> 5) & 7u)
      << ",\"wait_barrier_bits\":" << ((control >> 11) & 0x3fu)
      << "}"
      << ",\"control_decode_source\":\"NVBit_getSassBinary_raw_high_word\"";
}

inline void WriteOperand(std::ostream &out,
                         const Instruction::Operand &operand) {
  const bool has_legacy_immediate =
      operand.has_u64 || operand.has_f64 ||
      (operand.has_cbank && operand.cbank_has_imm) ||
      (operand.has_mref && operand.mref_has_imm);
  const int legacy_reg = operand.has_reg ? operand.reg_num : 0;
  out << "{\"text\":\"" << JsonEscape(operand.text)
      << "\",\"type\":\"" << JsonEscape(operand.type)
      << "\",\"is_negative\":"
      << (operand.is_negative ? "true" : "false")
      << ",\"is_not\":" << (operand.is_not ? "true" : "false")
      << ",\"is_absolute\":"
      << (operand.is_absolute ? "true" : "false")
      << ",\"nbytes\":" << operand.nbytes
      // Names mirror traced_operand::Serialize for adapter reuse.
      << ",\"operand_string\":\"" << JsonEscape(operand.text)
      << "\",\"operand_type\":\"" << JsonEscape(operand.type)
      << "\",\"has_reg\":" << (operand.has_reg ? "true" : "false")
      << ",\"legacy_reg_alias_ambiguous\":"
      << (operand.legacy_reg_alias_ambiguous ? "true" : "false")
      << ",\"has_pred\":" << (operand.has_pred ? "true" : "false")
      << ",\"has_mref\":" << (operand.has_mref ? "true" : "false")
      << ",\"is_negative_reg\":"
      << (operand.is_negative ? "true" : "false")
      << ",\"is_absolute_reg\":"
      << (operand.is_absolute ? "true" : "false")
      << ",\"operand_reg_number\":" << legacy_reg
      << ",\"has_inmediate\":"
      << (has_legacy_immediate ? "true" : "false")
      << ",\"operand_inmmediates\":[";
  bool wrote_immediate = false;
  if (operand.has_u64) {
    out << operand.u64;
    wrote_immediate = true;
  }
  if (operand.has_f64) {
    if (wrote_immediate) out << ',';
    if (std::isfinite(operand.f64)) out << std::setprecision(17) << operand.f64;
    else out << "null";
    wrote_immediate = true;
  }
  if (operand.has_cbank && operand.cbank_has_imm) {
    if (wrote_immediate) out << ',';
    out << operand.cbank_imm;
    wrote_immediate = true;
  }
  if (operand.has_mref && operand.mref_has_imm) {
    if (wrote_immediate) out << ',';
    out << operand.mref_imm;
  }
  out << "]"
      << ",\"is_reuse_bit_set\":" << (operand.reuse_bit ? "true" : "false")
      << ",\"operand_modifiers\":[";
  for (size_t modifier_index = 0; modifier_index < operand.modifiers.size();
       ++modifier_index) {
    if (modifier_index) out << ',';
    out << "\"" << JsonEscape(operand.modifiers[modifier_index]) << "\"";
  }
  out << "]"
      << ",\"register_refs\":[";
  for (size_t ref_index = 0; ref_index < operand.register_refs.size();
       ++ref_index) {
    if (ref_index) out << ',';
    const auto &ref = operand.register_refs[ref_index];
    out << "{\"role\":\"" << JsonEscape(ref.role)
        << "\",\"number\":" << ref.number
        << ",\"has_modifier\":"
        << (ref.has_modifier ? "true" : "false");
    if (ref.has_modifier)
      out << ",\"modifier\":\"" << JsonEscape(ref.modifier) << "\"";
    out << "}";
  }
  out << "]";
  if (operand.has_u64) out << ",\"immediate_u64\":" << operand.u64;
  if (operand.has_f64) {
    if (std::isfinite(operand.f64)) {
      out << ",\"immediate_f64\":" << std::setprecision(17) << operand.f64;
    } else {
      out << ",\"immediate_f64_nonfinite\":true";
    }
  }
  if (operand.has_reg) {
    out << ",\"reg_num\":" << operand.reg_num
        << ",\"has_reg_property\":"
        << (operand.has_reg_property ? "true" : "false");
    if (operand.has_reg_property)
      out << ",\"reg_property\":\""
          << JsonEscape(operand.reg_property) << "\"";
  }
  if (operand.has_pred) out << ",\"pred_num\":" << operand.pred_num;
  if (operand.has_cbank) {
    out << ",\"cbank\":{\"id\":" << operand.cbank_id
        << ",\"has_imm_offset\":"
        << (operand.cbank_has_imm ? "true" : "false")
        << ",\"imm_offset\":" << operand.cbank_imm
        << ",\"has_reg_offset\":"
        << (operand.cbank_has_reg ? "true" : "false")
        << ",\"reg_offset\":" << operand.cbank_reg << "}";
  }
  if (operand.has_mref) {
    out << ",\"mref\":{\"has_ra\":"
        << (operand.mref_has_ra ? "true" : "false")
        << ",\"ra_num\":" << operand.mref_ra
        << ",\"ra_modifier\":\""
        << JsonEscape(operand.mref_ra_modifier) << "\""
        << ",\"has_ur\":" << (operand.mref_has_ur ? "true" : "false")
        << ",\"ur_num\":" << operand.mref_ur
        << ",\"ur_modifier\":\""
        << JsonEscape(operand.mref_ur_modifier) << "\""
        << ",\"has_imm\":" << (operand.mref_has_imm ? "true" : "false")
        << ",\"imm\":" << operand.mref_imm
        << ",\"has_desc\":" << (operand.mref_has_desc ? "true" : "false")
        << ",\"desc_ureg\":" << operand.mref_desc << "}";
  }
  if (operand.has_generic)
    out << ",\"generic\":\"" << JsonEscape(operand.generic_text) << "\"";
  if (operand.has_desc) out << ",\"desc_ureg\":" << operand.desc_ureg;
  if (operand.has_tma) {
    out << ",\"tma_ureg\":[";
    for (int i = 0; i < 4; ++i) {
      if (i) out << ',';
      out << operand.tma_ureg[i];
    }
    out << "]";
  }
  out << "}";
}

inline bool WriteExecutionInfo(const std::string &path,
                               const std::string &benchmark_name,
                               const std::map<uint32_t, Function> &functions) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) return false;
  out << "{\n  \"benchmark_name\":\"" << JsonEscape(benchmark_name)
      << "\",\n  \"metadata_kind\":\"native_static_sm120_v3\""
      << ",\"static_exporter_version\":3"
      << ",\"static_source\":\"NVBit Instr::getSass/getSassBinary/getOperand\""
      << ",\"control_latency_claims\":false"
      << ",\"kernels\":[\n";
  bool first_function = true;
  for (const auto &entry : functions) {
    const Function &fn = entry.second;
    if (!first_function) out << ",\n";
    first_function = false;
    out << "    {\"kernel_name\":\"" << JsonEscape(fn.name)
        << "\",\"unique_function_id\":" << fn.function_id
        << ",\"architecture_version\":" << fn.architecture_version
        << ",\"function_address\":" << fn.function_address
        << ",\"is_captured_from_binary\":true"
        << ",\"instructions\":[\n";
    for (size_t i = 0; i < fn.instructions.size(); ++i) {
      const Instruction &inst = fn.instructions[i];
      if (i) out << ",\n";
      out << "      {\"pc_string_hex\":\"" << std::hex << inst.pc
        << std::dec << "\",\"pc_num_dec\":" << inst.pc
          << ",\"op_code\":\"" << JsonEscape(inst.opcode) << "\""
          << ",\"sass\":\"" << JsonEscape(inst.sass) << "\""
          << ",\"opcode\":\"" << JsonEscape(inst.opcode)
          << "\",\"opcode_short\":\"" << JsonEscape(inst.opcode_short)
          << "\",\"is_predicated\":" << (inst.predicated ? "true" : "false")
          << ",\"is_uniform_predicate\":"
          << (inst.predicate_uniform ? "true" : "false")
          << ",\"is_predicate_negate\":"
          << (inst.predicate_negated ? "true" : "false")
          << ",\"predicate_register\":" << inst.predicate_register
          << ",\"predicate_register_available\":"
          << (inst.predicate_register_available ? "true" : "false")
          << ",\"encoding_byte_count\":" << inst.encoding.size()
          << ",\"encoding_hex\":\"" << HexBytes(inst.encoding, true)
          << "\",\"encoding_bytes_forward_hex\":\""
          << HexBytes(inst.encoding, false) << "\""
          << ",\"word0_hex\":\"" << HexWord(inst.encoding, 0, 8)
          << "\",\"word1_hex\":\"" << HexWord(inst.encoding, 8, 8)
          << "\""
          << ",\"operands\":[";
      for (size_t operand_index = 0; operand_index < inst.operands.size();
           ++operand_index) {
        if (operand_index) out << ',';
        WriteOperand(out, inst.operands[operand_index]);
      }
      out << "]"
          << ",\"register_metadata_available\":false"
          << ",\"register_metadata_unavailable_reason\":\"NVBit_Instr_API_does_not_expose_RFU\"";
      WriteControlFields(out, inst.encoding, fn.architecture_version);
      out << "}";
    }
    out << "\n    ]}";
  }
  out << "\n  ]\n}\n";
  out.flush();
  return static_cast<bool>(out);
}

}  // namespace nvbit_static_metadata
