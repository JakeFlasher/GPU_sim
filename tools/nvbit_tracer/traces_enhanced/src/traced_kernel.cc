// Copyright (c) 2023-2025, Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "traced_kernel.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include "traced_operand.h"
#include "string_utilities.h"

namespace {

// NVBit/cuobjdump prefixes use @P0..@P6 and @PT (or the corresponding UP
// form; PT is predicate index 7). Parse the complete token here so malformed
// prefixes cannot be silently reinterpreted as another predicate or escape
// through stoul().
bool parse_predicate_token(const std::string &token, bool &is_negated,
                           bool &is_uniform, unsigned int &predicate) noexcept {
  if (token.empty() || token.front() != '@') {
    return false;
  }

  std::size_t index = 1;
  is_negated = false;
  is_uniform = false;
  if (index < token.size() && token[index] == '!') {
    is_negated = true;
    ++index;
  }
  if (index < token.size() && token[index] == 'U') {
    is_uniform = true;
    ++index;
  }
  if (index >= token.size() || token[index] != 'P') {
    return false;
  }
  ++index;
  if (index + 1 != token.size()) {
    return false;
  }

  const char register_char = token[index];
  if (register_char == 'T') {
    predicate = is_uniform ? UPT : PT;
    return true;
  }
  if (register_char < '0' || register_char > '6') {
    return false;
  }
  predicate = static_cast<unsigned int>(register_char - '0');
  return true;
}

}  // namespace

bool track_this_instruction(std::size_t icount, int threshold_unique_kernel_checking, std::string opcode_str) {
  bool res = threshold_unique_kernel_checking > 0 &&
      icount < static_cast<std::size_t>(threshold_unique_kernel_checking) &&
      (opcode_str.find("BRA") == std::string::npos) && (opcode_str.find("BSSY") == std::string::npos) && (opcode_str.find("CALL") == std::string::npos) && (opcode_str.find("RET") == std::string::npos) && (opcode_str.find("MOV") == std::string::npos);
  return res;
}

std::shared_ptr<traced_instruction> create_no_binay_instruction(unsigned int pc, std::string instruction_str) {
    unsigned int  next_idx_to_check = 0;
    bool is_predicated = false;
    bool is_predicate_negate = false;
    bool is_uniform_predicate = false;
    unsigned int predicate_register = 0;
    std::vector<std::string> stripped_inst = split_string(instruction_str, ' ');
    if (stripped_inst.empty()) {
        return nullptr;
    }
    if(stripped_inst[next_idx_to_check][0] == '@') {
        if (!parse_predicate_token(stripped_inst[next_idx_to_check],
                                   is_predicate_negate,
                                   is_uniform_predicate,
                                   predicate_register)) {
            return nullptr;
        }
        is_predicated = true;
        next_idx_to_check++;
    }
    if (next_idx_to_check >= stripped_inst.size() ||
        stripped_inst[next_idx_to_check].empty()) {
        return nullptr;
    }
    std::string op_code = stripped_inst[next_idx_to_check];
    std::stringstream streamStrPC;
    streamStrPC << std::hex << pc;
    std::string pc_string( streamStrPC.str() );
    try {
      std::shared_ptr<traced_instruction> instruction =
          std::make_shared<traced_instruction>(
              pc_string, pc, op_code, is_predicated, is_uniform_predicate,
              is_predicate_negate, predicate_register);
    next_idx_to_check++;

    // Create operands
    for(unsigned i = next_idx_to_check; i < stripped_inst.size(); i++) {
        if(!stripped_inst[i].empty() && (stripped_inst[i][0]!='&') && (stripped_inst[i][0]!='?') && (stripped_inst[i][0]!=';')) {
            instruction->add_operand(std::make_unique<traced_operand>(stripped_inst[i]));
        }
    }
    instruction->calculate_num_destination_registers();
    return instruction;
    } catch (const std::exception &) {
      return nullptr;
    }
}

traced_kernel::traced_kernel(std::string kernel_name, unsigned int architecture_version) {
    m_kernel_name = kernel_name;
    m_architecture_version = architecture_version;
    m_unique_function_id = 0;
    m_func_addr = 0;
    m_is_captured_from_binary = false;
    m_last_parsed_pc = 0;
    m_OpcodeMap = nullptr;
}

traced_kernel::traced_kernel()
    : traced_kernel("", 0) {}

traced_kernel::~traced_kernel() {
    m_instructions.clear();
    m_key_instructions_pcs.clear();
}

void traced_kernel::set_unique_function_id(unsigned int unique_function_id) {
    m_unique_function_id = unique_function_id;
}

unsigned int traced_kernel::get_unique_function_id() {
    return m_unique_function_id;
}

uint64_t traced_kernel::get_function_addr() {
    return m_func_addr;
}

void traced_kernel::set_function_addr(uint64_t function_addr) {
    m_func_addr = function_addr;
}

bool traced_kernel::is_captured_from_binary() const {
    return m_is_captured_from_binary;
}

void traced_kernel::set_is_captured_from_binary(bool is_captured_from_binary) {
    m_is_captured_from_binary = is_captured_from_binary;
}

bool traced_kernel::add_instruction(std::vector<std::string> instruction_part1, std::vector<std::string> instruction_part2, int threshold_unique_kernel_checking, std::string full_instruction_str) {
    if (instruction_part1.empty()) {
        return false;
    }
    try {
    if(instruction_part1.size() > 1) {
        std::string pc_string = instruction_part1[0];
        std::size_t parsed = 0;
        unsigned long parsed_pc = std::stoul(pc_string, &parsed, 16);
        if (parsed != pc_string.size() ||
            parsed_pc > std::numeric_limits<unsigned int>::max() ||
            parsed_pc > static_cast<unsigned long>(std::numeric_limits<int>::max())) {
            return false;
        }
        unsigned int pc_num = static_cast<unsigned int>(parsed_pc);
        bool already_exists = m_instructions.find(pc_num) != m_instructions.end();
        if (already_exists || instruction_part1[1].empty()) {
            return false;
        }
        unsigned int  next_idx_to_check = 1;
        bool is_predicated = false;
        bool is_predicate_negate = false;
        bool is_uniform_predicate = false;
        unsigned int predicate_register = 0;
            if(instruction_part1[next_idx_to_check][0] == '@') {
            if (!parse_predicate_token(instruction_part1[next_idx_to_check],
                                       is_predicate_negate,
                                       is_uniform_predicate,
                                       predicate_register)) {
                return false;
            }
            is_predicated = true;
            next_idx_to_check++;
        }

        if (next_idx_to_check >= instruction_part1.size() - 1 ||
            instruction_part1.back().empty()) {
            return false;
        }
        std::string op_code = instruction_part1[next_idx_to_check];
        std::string tracked_sass_instruction;
        if(track_this_instruction(m_instructions.size(), threshold_unique_kernel_checking, op_code)) {
            tracked_sass_instruction = create_sass_instr(full_instruction_str, false, "");
            if (tracked_sass_instruction.empty()) {
                return false;
            }
        }
        next_idx_to_check++;

        std::vector<std::string> encoded_instruction;
        encoded_instruction.push_back(instruction_part1[instruction_part1.size() - 1]);
        if(m_architecture_version >= 70) {
            if (instruction_part2.empty() || instruction_part2[0].empty()) {
                return false;
            }
            encoded_instruction.push_back(instruction_part2[0]);
        }
        auto instruction = std::make_shared<traced_instruction>(
            pc_string, pc_num, op_code, encoded_instruction, is_predicated,
            is_uniform_predicate, is_predicate_negate, predicate_register);
        
        // Create operands
        for(unsigned i = next_idx_to_check; (i < instruction_part1.size() -1); i++) {
            if(!instruction_part1[i].empty() && (instruction_part1[i][0]!='&') && (instruction_part1[i][0]!='?') && (instruction_part1[i][0]!=';')) {
                instruction->add_operand(std::make_unique<traced_operand>(instruction_part1[i]));
            }
        }

        instruction->calculate_num_destination_registers();
        auto [instruction_it, inserted] =
            m_instructions.emplace(pc_num, std::move(instruction));
        if (!inserted) {
            return false;
        }
        try {
            if (!tracked_sass_instruction.empty()) {
                m_key_instructions_pcs.insert_or_assign(
                    static_cast<int>(pc_num), std::move(tracked_sass_instruction));
            }
        } catch (...) {
            m_instructions.erase(instruction_it);
            throw;
        }
        m_last_parsed_pc = std::max(m_last_parsed_pc, pc_num);
        return true;
    }else {
        if (m_last_parsed_pc >
            static_cast<unsigned int>(std::numeric_limits<int>::max()) - 16U) {
            return false;
        }
        unsigned int candidate_pc = m_last_parsed_pc + 16;
        if (m_instructions.find(candidate_pc) != m_instructions.end()) {
            return false;
        }
        std::string pc_str = decimalToHexString(candidate_pc, 4);
        const bool inserted = m_instructions.emplace(
            candidate_pc,
            std::make_shared<traced_instruction>(
                pc_str, candidate_pc, "NOP", false, false, false, 0)).second;
        if (!inserted) {
            return false;
        }
        m_last_parsed_pc = candidate_pc;
        return true;
    }
    } catch (const std::exception &) {
        return false;
    }
}

bool traced_kernel::add_no_binary_instruction(unsigned int pc, std::string instruction_str) {
    bool already_exists = m_instructions.find(pc) != m_instructions.end();
    if (already_exists) {
        return false;
    }
    auto instruction = create_no_binay_instruction(pc, instruction_str);
    if (instruction == nullptr) {
        return false;
    }
    const bool inserted = m_instructions.emplace(pc, std::move(instruction)).second;
    if (!inserted) {
        return false;
    }
    m_last_parsed_pc = std::max(m_last_parsed_pc, pc);
    return true;
}

bool traced_kernel::add_register_usage(unsigned pc, std::string reg_file_name, unsigned num_regs) {
    auto it = m_instructions.find(pc);
    if (it == m_instructions.end()) {
        return false;
    }
    return it->second->add_register_usage(reg_file_name, num_regs);
}

bool traced_kernel::add_call_target(unsigned pc, std::string target_name, int target_id) {
    auto it = m_instructions.find(pc);
    if (it == m_instructions.end()) {
        return false;
    }
    return it->second->add_call_target(target_name, target_id);
}

std::string traced_kernel::get_kernel_name() const {
    return m_kernel_name;
}

traced_instruction& traced_kernel::get_instruction(unsigned pc) {
    return *m_instructions.at(pc);
}

std::shared_ptr<traced_instruction> traced_kernel::get_instruction_ptr(unsigned pc) {
    if (m_instructions.find(pc) != m_instructions.end())
    {
        return m_instructions.at(pc);
    }
    else
    {
        return std::shared_ptr<traced_instruction>(nullptr);
    }
}

bool traced_kernel::is_kernel_in_use() const {
    return m_unique_function_id != 0;
}

std::map<int, std::string> traced_kernel::get_key_instructions_pcs() const {
    return m_key_instructions_pcs;
}

void traced_kernel::set_kernel_name(std::string kernel_name) {
    m_kernel_name = kernel_name;
}

bool traced_kernel::set_opcode_map() {
    return get_opcode_map(m_OpcodeMap, m_architecture_version);
}

bool traced_kernel::Serialize(rapidjson::Writer<rapidjson::StringBuffer> * writer) const {
    writer->StartObject();

    writer->String("kernel_name"); 
    writer->String(m_kernel_name.c_str());

    writer->String("architecture_version"); 
    writer->Int64(m_architecture_version);

    writer->String("is_captured_from_binary");
    writer->Bool(m_is_captured_from_binary);

    writer->String("unique_function_id");
    writer->Int64(m_unique_function_id);

    writer->String("function_address");
    writer->Uint64(m_func_addr);

    std::vector<unsigned> keys_pc;
    for(auto it = m_instructions.begin(); it != m_instructions.end(); it++) {
        keys_pc.push_back(it->first);
    }
    std::sort(keys_pc.begin(), keys_pc.end());
    writer->String("instructions"); 
    writer->StartArray();
    for(auto it = keys_pc.begin(); it != keys_pc.end(); it++) {
        m_instructions.at(*it)->Serialize(writer);
    }
    writer->EndArray();

    writer->EndObject();

    return true;
}


bool traced_kernel::Deserialize(const rapidjson::Value & obj) {
    std::string kernel_name;
    if (!obj.IsObject() || !HasUniqueObjectMembers(obj) ||
        !obj.HasMember("kernel_name") ||
        !ReadString(obj["kernel_name"], kernel_name) ||
        !obj.HasMember("architecture_version") ||
        !obj["architecture_version"].IsInt64() ||
        obj["architecture_version"].GetInt64() < 0 ||
        static_cast<uint64_t>(obj["architecture_version"].GetInt64()) >
            std::numeric_limits<unsigned int>::max() ||
        !obj.HasMember("is_captured_from_binary") ||
        !obj["is_captured_from_binary"].IsBool() ||
        !obj.HasMember("unique_function_id") ||
        !obj["unique_function_id"].IsInt64() ||
        obj["unique_function_id"].GetInt64() < 0 ||
        static_cast<uint64_t>(obj["unique_function_id"].GetInt64()) >
            std::numeric_limits<unsigned int>::max() ||
        !obj.HasMember("function_address") ||
        !obj["function_address"].IsUint64() ||
        !obj.HasMember("instructions") || !obj["instructions"].IsArray()) {
        return false;
    }
    const unsigned int architecture_version =
        static_cast<unsigned int>(obj["architecture_version"].GetInt64());
    const bool is_captured_from_binary = obj["is_captured_from_binary"].GetBool();
    const unsigned int unique_function_id =
        static_cast<unsigned int>(obj["unique_function_id"].GetInt64());
    const uint64_t function_address = obj["function_address"].GetUint64();

    // Build the complete candidate state off to the side.  A failed nested
    // instruction must leave an already-published kernel usable; mutating the
    // live maps before the final record is validated otherwise leaves a
    // partially decoded kernel behind for a later retry.
    const std::unordered_map<std::string, OpcodeChar> *opcode_map = nullptr;
    if (!get_opcode_map(opcode_map, architecture_version)) {
        return false;
    }
    std::map<unsigned int, std::shared_ptr<traced_instruction>> parsed_instructions;
    const rapidjson::Value &instructions = obj["instructions"];
    for (rapidjson::SizeType i = 0; i < instructions.Size(); i++)
    {
        std::shared_ptr<traced_instruction> instruction = std::make_shared<traced_instruction>();
        if (!instruction->Deserialize(instructions[i])) {
            return false;
        }
        if (is_captured_from_binary &&
            instructions[i]["encoded_instruction"].Size() == 0) {
            // Binary-backed metadata must retain its encoded instruction words;
            // an empty vector is reserved for the explicit no-binary fallback.
            return false;
        }
        instruction->set_simulation_opcode(opcode_map, instruction->get_op_code());
        unsigned pc_num = 0;
        try {
            std::size_t parsed = 0;
            unsigned long value = std::stoul(instruction->get_pc_string(), &parsed, 16);
            if (parsed != instruction->get_pc_string().size() ||
                value > std::numeric_limits<unsigned int>::max()) {
                return false;
            }
            pc_num = static_cast<unsigned int>(value);
        } catch (const std::exception &) {
            return false;
        }
        if (parsed_instructions.find(pc_num) != parsed_instructions.end()) {
            return false;
        }
        parsed_instructions[pc_num] = std::move(instruction);
    }

    const unsigned int last_parsed_pc =
        parsed_instructions.empty() ? 0 : parsed_instructions.rbegin()->first;

    m_instructions = std::move(parsed_instructions);
    m_key_instructions_pcs.clear();
    m_kernel_name = kernel_name;
    m_architecture_version = architecture_version;
    m_is_captured_from_binary = is_captured_from_binary;
    m_unique_function_id = unique_function_id;
    m_func_addr = function_address;
    m_last_parsed_pc = last_parsed_pc;
    m_OpcodeMap = opcode_map;
  return true;
}
