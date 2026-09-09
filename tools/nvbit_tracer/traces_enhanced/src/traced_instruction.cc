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

#include "traced_instruction.h"
#include "string_utilities.h"

// NOTE: the old upstream trace_parser.h include was removed here.
// Verified: no symbols from trace_parser.h are used anywhere in traces_enhanced/src/.
// This isolates the host-built tracer from unrelated upstream headers.
#include "../../isa_def/ampere_opcode.h"
#include "../../isa_def/hopper_opcode.h"
#include "../../isa_def/blackwell_opcode.h"
#include "../../isa_def/pascal_opcode.h"
#include "../../isa_def/turing_opcode.h"
#include "../../isa_def/volta_opcode.h"
#include "../../isa_def/kepler_opcode.h"

#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <utility>

bool get_opcode_map(const std::unordered_map<std::string, OpcodeChar> *&OpcodeMap, int binary_verion) {
  OpcodeMap = nullptr;
  if(binary_verion == BLACKWELL_RTX_BINART_VERSION)
    OpcodeMap = &Blackwell_OpcodeMap;
  else if (binary_verion == ADA_BINART_VERSION)
    OpcodeMap = &Ampere_OpcodeMap;
  else if (binary_verion == HOPPER_RTX_BINART_VERSION)
    OpcodeMap = &Hopper_OpcodeMap;
  else if (binary_verion == AMPERE_RTX_BINART_VERSION ||
      binary_verion == AMPERE_A100_BINART_VERSION)
    OpcodeMap = &Ampere_OpcodeMap;
  else if (binary_verion == VOLTA_BINART_VERSION ||
           binary_verion == VOLTA_JETSON_BINART_VERSION)
    OpcodeMap = &Volta_OpcodeMap;
  else if (binary_verion == PASCAL_TITANX_BINART_VERSION ||
           binary_verion == PASCAL_P100_BINART_VERSION)
    OpcodeMap = &Pascal_OpcodeMap;
  else if (binary_verion == KEPLER_BINART_VERSION)
    OpcodeMap = &Kepler_OpcodeMap;
  else if (binary_verion == TURING_BINART_VERSION)
    OpcodeMap = &Turing_OpcodeMap;
  else {
    printf("unsupported binary version: %d\n",
           binary_verion);
    fflush(stdout);
    return false;
  }
  return true;
}

bool is_reserved_reg(unsigned int reg_id, TraceEnhancedOperandType reg_type) {
  bool res = false;
  if(reg_type == TraceEnhancedOperandType::REG) {
    res = (reg_id == RZ);
  } else if(reg_type == TraceEnhancedOperandType::UREG) {
    res = (reg_id == URZ);
  } else if(reg_type == TraceEnhancedOperandType::PRED) {
    res = (reg_id == PT);
  } else if(reg_type == TraceEnhancedOperandType::UPRED) {
    res = (reg_id == UPT);
  }
  return res;
}

void advance_operand_idx(unsigned int &operand_idx, traced_instruction &static_inst_info) {
  while((operand_idx < static_inst_info.get_num_operands()) && !has_the_operand_type_been_captured(static_inst_info.get_operand(operand_idx).get_operand_type())) {
    operand_idx++;
  }
}

bool has_the_operand_type_been_captured(TraceEnhancedOperandType reg_type) {
  return (reg_type == TraceEnhancedOperandType::REG) || (reg_type == TraceEnhancedOperandType::UREG) || (reg_type == TraceEnhancedOperandType::PRED) ||
   (reg_type == TraceEnhancedOperandType::UPRED);
}

unsigned int get_number_of_uses_per_operand(traced_instruction &trace_inst, unsigned int reg_id, unsigned int operand_pos, TraceEnhancedOperandType reg_type) {
  unsigned int num_uses = 1;
  if(!is_reserved_reg(reg_id, reg_type)) {
    if(reg_type == TraceEnhancedOperandType::REG && (trace_inst.is_dp_op())) {
      if((trace_inst.get_op_code().find("DMMA") != std::string::npos) && ( (operand_pos == 0) || (operand_pos == 3) ) ) {
        num_uses = 4;
      }else if(trace_inst.get_op_code().find("F2F") != std::string::npos) {
        std::vector<std::string> op_code_tokens = get_opcode_tokens(trace_inst.get_op_code());
        if(op_code_tokens.size() > (operand_pos + 1)) {
          if(op_code_tokens[operand_pos + 1].find("64") != std::string::npos) {
            num_uses = 2;
          }
        }
      }else if(reg_id % 2 == 0) {
        num_uses = 2;
      }
    }else if(reg_type == TraceEnhancedOperandType::PRED && (reg_id == PR)) {
      num_uses = PR;
    }else if(reg_type == TraceEnhancedOperandType::UPRED && (reg_id == UPR)) {
      num_uses = UPR;
    }else if(trace_inst.get_is_imad_wide() && (operand_pos == 0) ) {
      num_uses = 2;
    }else if( trace_inst.is_load_op() && (operand_pos == 1) && (trace_inst.get_op_code().find(".E") != std::string::npos ) ) {
      num_uses = 2;
    }else if( trace_inst.is_store_op() && (operand_pos == 0) && (trace_inst.get_op_code().find(".E") != std::string::npos ) ) {
      num_uses = 2;
    }else if( (operand_pos == 0) && (trace_inst.get_op_code().find("LD") != std::string::npos ) ) {
      if( (trace_inst.get_op_code().find("64") != std::string::npos ) ){
        num_uses = 2;
      }else if( (trace_inst.get_op_code().find("128") != std::string::npos) && (trace_inst.get_op_code().find("LTC128B") == std::string::npos) ){
        num_uses = 4;
      }else if( (trace_inst.get_op_code().find("LDSM") != std::string::npos ) ){
        if( (trace_inst.get_op_code().find(".2") != std::string::npos ) ){
          num_uses = 2;
        }else if( (trace_inst.get_op_code().find(".4") != std::string::npos ) ){
          num_uses = 4;
        }
      }
    }else if( (operand_pos == 1) && (trace_inst.get_op_code().find("ST") != std::string::npos ) ) {
      if( (trace_inst.get_op_code().find("64") != std::string::npos ) ){
        num_uses = 2;
      }else if( (trace_inst.get_op_code().find("128") != std::string::npos ) ){
        num_uses = 4;
      }
    }else if(trace_inst.is_tensor_core_op()) {
      if (!trace_inst.get_tensor_core_instruction_info().is_set) {
        throw std::logic_error(
            "tensor-core register demand requested without shape metadata");
      }
      const Tensor_core_instruction_information& tensor_info =
          trace_inst.get_tensor_core_instruction_info();
      const bool integer_tensor =
          trace_inst.get_op_code().find("BMMA") != std::string::npos ||
          trace_inst.get_op_code().find("IMMA") != std::string::npos;
      if((operand_pos == 0) || (operand_pos == 3)) {
        unsigned int operand_bit_size = tensor_info.operand_bit_size;
        if (integer_tensor) {
          operand_bit_size = 32;
        }
        num_uses = (tensor_info.size_m * tensor_info.size_n * operand_bit_size) / 1024;
        
      }else if(operand_pos == 1) {
        num_uses = (tensor_info.size_m * tensor_info.size_k * tensor_info.operand_bit_size) / 1024;
        // HMMA floating-point fragments use packed 16-bit values in .b32
        // registers, so the historical factor of two converts element
        // demand to register demand. Dense IMMA/BMMA integer fragments are
        // already represented in the PTX element width used above (for
        // example, two .b32 A registers for m16n8k16.s8). Sparse integer
        // instructions retain half of A, so preserve that reduction only for
        // the sparse A fragment.
        if (!integer_tensor ||
            tensor_info.is_sparse) {
          num_uses = num_uses / 2;
        }
      }else if(operand_pos == 2) {
        num_uses = (tensor_info.size_k * tensor_info.size_n * tensor_info.operand_bit_size) / 1024;
        if (!integer_tensor) {
          num_uses = num_uses / 2;
        }
      }
    }else if( (trace_inst.get_op_code().find("F2") != std::string::npos) || (trace_inst.get_op_code().find("2F") != std::string::npos)) {
      std::size_t size64pos = trace_inst.get_op_code().find("64");
      std::size_t size32pos = trace_inst.get_op_code().find("32");
      std::size_t size16pos = trace_inst.get_op_code().find("16");
      std::size_t size8pos = trace_inst.get_op_code().find("8");
      bool is_64 = (size64pos != std::string::npos);
      if(is_64) {
        if( (operand_pos == 0 )&& (size64pos < size32pos) && (size64pos < size16pos) && (size64pos < size8pos) ) {
          num_uses = 2;
        }else if( (operand_pos == 1 ) && ((size64pos > size32pos) || (size64pos > size16pos) || (size64pos > size8pos)) ) {
          num_uses = 2;
        }
      }
    }
    if(num_uses == 2) { // Final check
      num_uses = (reg_id % 2 == 0) ? 2 : 1;
    }else if(num_uses == 4) {
      // SM120 QMMA may use non-4-aligned register IDs for multi-register
      // operands. Clamp to the largest power-of-2 alignment available.
      if(reg_id % 4 != 0) {
        num_uses = (reg_id % 2 == 0) ? 2 : 1;
      }
    }
  }
  return num_uses;
}

s2r_operand_type get_s2r_operand_type(std::string operand_string) {
  s2r_operand_type operand_type = NONE_S2R_OPERAND_TYPE;
  if(operand_string.find("CTAID.X") != std::string::npos) {
    operand_type = CTA_ID_X_S2R_OPERAND_TYPE;
  }else if(operand_string.find("CTAID.Y") != std::string::npos) {
    operand_type = CTA_ID_Y_S2R_OPERAND_TYPE;
  }else if(operand_string.find("CTAID.Z") != std::string::npos) {
    operand_type = CTA_ID_Z_S2R_OPERAND_TYPE;
  }else if(operand_string.find("TID.X") != std::string::npos) {
    operand_type = THREAD_ID_X_S2R_OPERAND_TYPE;
  }else if(operand_string.find("TID.Y") != std::string::npos) {
    operand_type = THREAD_ID_Y_S2R_OPERAND_TYPE;
  }else if(operand_string.find("TID.Z") != std::string::npos) {
    operand_type = THREAD_ID_Z_S2R_OPERAND_TYPE;
  }else if(operand_string.find("LANEID") != std::string::npos) {
    operand_type = LANE_ID_S2R_OPERAND_TYPE;
  }else if(operand_string.find("LTMASK") != std::string::npos) {
    operand_type = LTMASK_S2R_OPERAND_TYPE;
  }
  return operand_type;
}

traced_instruction::traced_instruction(std::string pc_string, unsigned pc_num, std::string op_code, std::vector<std::string> encoded_instruction, bool is_predicated, bool is_uniform_predicate, bool is_predicate_negate, unsigned predicate_register) {
    m_pc_string = pc_string;
    m_pc_num = pc_num;
    m_is_imad = false;
    m_is_imad_wide = false;
    m_contains_setp = false;
    m_membar_type = MEMBAR_TRACED_NONE;
    set_op_code(op_code);
    m_encoded_instruction = encoded_instruction;
    m_is_predicated = is_predicated;
    m_is_uniform_predicate = is_uniform_predicate;
    m_is_predicate_negate = is_predicate_negate;
    m_predicate_register = predicate_register;
    m_register_usage = std::make_unique<register_usage>(0, 0, 0, 0);
    if (encoded_instruction.size() >= 2) {
        m_control_bits = std::make_unique<control_bits>(
            encoded_instruction[1], CCPos_arch_7x_8x);
    } else {
        // Pre-Volta SASS carries no second control-word token in the
        // cuobjdump output consumed by this parser. Preserve the encoded
        // instruction word but expose an explicit zero/default control block
        // instead of indexing past the vector in release builds.
        m_control_bits = std::make_unique<control_bits>();
    }
    m_simulation_opcode = NO_OP;
    m_num_destination_registers = 0;
}

traced_instruction::traced_instruction(std::string pc_string, unsigned pc_num, std::string op_code, bool is_predicated, bool is_uniform_predicate, bool is_predicate_negate, unsigned predicate_register) {
    m_pc_string = pc_string;
    m_pc_num = pc_num;
    m_is_imad = false;
    m_is_imad_wide = false;
    m_membar_type = MEMBAR_TRACED_NONE;
    m_contains_setp = false;
    set_op_code(op_code);
    m_encoded_instruction = std::vector<std::string>();
    m_is_predicated = is_predicated;
    m_is_uniform_predicate = is_uniform_predicate;
    m_is_predicate_negate = is_predicate_negate;
    m_predicate_register = predicate_register;
    m_register_usage = std::make_unique<register_usage>(0, 0, 0, 0);
    m_control_bits = std::make_unique<control_bits>();
    m_simulation_opcode = NO_OP;
    m_num_destination_registers = 0;
}

traced_instruction::traced_instruction()
    : traced_instruction(
          "", 0, "NOP", std::vector<std::string>{}, false, false, false, 0) {}

traced_instruction::~traced_instruction() {
    m_operands.clear();
}

void traced_instruction::add_operand(std::unique_ptr<traced_operand> operand) {
    if (operand == nullptr) {
        throw std::invalid_argument("cannot attach a null traced operand");
    }
    m_operands.push_back(std::move(operand));
}

std::vector<std::unique_ptr<traced_operand>> &traced_instruction::get_operands() {
    return m_operands;
}

std::size_t traced_instruction::get_num_operands() {
    return m_operands.size();
}

traced_operand& traced_instruction::get_operand(unsigned int idx) {
    return *m_operands.at(idx);
}

bool traced_instruction::get_is_imad() {
    return m_is_imad;
}

bool traced_instruction::get_is_imad_wide() {
    return m_is_imad_wide;
}

bool traced_instruction::get_is_system_memory_barrier() {
    return m_membar_type == MEMBAR_TRACED_SYS;
}

bool traced_instruction::get_is_cta_memory_barrier() {
    return m_membar_type == MEMBAR_TRACED_CTA;
}

bool traced_instruction::get_contains_setp() {
    return m_contains_setp;
}

bool traced_instruction::get_is_call_or_ret_with_relative() {
    return (m_op_code.find(std::string(".REL")) != std::string::npos);
}

std::string traced_instruction::get_op_code() {
    return m_op_code;
}

void traced_instruction::set_op_code(std::string op_code) {
    m_op_code = op_code;
    if(m_op_code.find(std::string("IMAD")) != std::string::npos) {
        m_is_imad = true;
        if(m_op_code.find(std::string("WIDE")) != std::string::npos) {
            m_is_imad_wide = true;
        }
    }else if(m_op_code.find(std::string("SETP")) != std::string::npos) {
        m_contains_setp = true;
    }else if(m_op_code.find(std::string("MEMBAR")) != std::string::npos) {
        if(m_op_code.find(std::string(".CTA")) != std::string::npos) {
            m_membar_type = MEMBAR_TRACED_CTA;
        }else if(m_op_code.find(std::string(".SYS")) != std::string::npos) {
            m_membar_type = MEMBAR_TRACED_SYS;
        }else if(m_op_code.find(std::string(".VC")) != std::string::npos) {
            m_membar_type = MEMBAR_TRACED_VC;
        }else {
            m_membar_type = MEMBAR_TRACED_GPU;
        }
    }
}

bool traced_instruction::add_register_usage(std::string reg_file_name, unsigned num_regs) {
    if(reg_file_name == std::string("UGPR")) {
        m_register_usage->set_num_uniform_registers(num_regs);
    }else if(reg_file_name == std::string("UPRED")) {
        m_register_usage->set_num_uniform_predicate_registers(num_regs);
    }else if(reg_file_name == std::string("GPR")) {
        m_register_usage->set_num_regular_registers(num_regs);
    }else if(reg_file_name == std::string("PRED")) {
        m_register_usage->set_num_regular_predicate_registers(num_regs);
    }else {
        return false;
    }
    return true;
}

bool traced_instruction::add_call_target(std::string target_name, int target_id) {
    m_operands.push_back(std::make_unique<traced_operand>(target_name, target_id));
    return true;
}

void traced_instruction::calculate_num_destination_registers() {
    if(m_operands.size() == 0) {
        m_num_destination_registers = 0;
    }else if( (m_operands[0]->get_operand_type() == TraceEnhancedOperandType::MREF) || 
        (m_operands[0]->get_operand_type() == TraceEnhancedOperandType::CBANK) || 
        (m_operands[0]->get_operand_type() == TraceEnhancedOperandType::DESC) ||
        (m_operands[0]->get_operand_type() == TraceEnhancedOperandType::IMM_DOUBLE) ||
        (m_operands[0]->get_operand_type() == TraceEnhancedOperandType::IMM_UINT64) ||
        (m_operands[0]->get_operand_type() == TraceEnhancedOperandType::SR) ||
        (m_operands[0]->get_operand_type() == TraceEnhancedOperandType::SB) ||
        (m_operands[0]->get_operand_type() == TraceEnhancedOperandType::GENERIC) ) {
        m_num_destination_registers = 0;
    }else if((m_op_code.find(std::string("BRA")) != std::string::npos) || 
        (m_op_code.find(std::string("BRX")) != std::string::npos) || 
        (m_op_code.find(std::string("BRXU")) != std::string::npos) || 
        (m_op_code.find(std::string("JMP")) != std::string::npos) || 
        (m_op_code.find(std::string("JMX")) != std::string::npos) || 
        (m_op_code.find(std::string("JMXU")) != std::string::npos) || 
        (m_op_code.find(std::string("BRK")) != std::string::npos) || 
        (m_op_code.find(std::string("CAL")) != std::string::npos) || 
        (m_op_code.find(std::string("CALL")) != std::string::npos) || 
        (m_op_code.find(std::string("WARPSYNC")) != std::string::npos) || 
        (m_op_code.find(std::string("YIELD")) != std::string::npos) || 
        (m_op_code.find(std::string("EXIT")) != std::string::npos) || 
        (m_op_code.find(std::string("RET")) != std::string::npos)) {
        m_num_destination_registers = 0;
    }else {
        m_num_destination_registers = 1; // However, this might change during simulation due to the tracking of modifiers.
    } 
}

unsigned int traced_instruction::get_num_destination_registers() {
    return m_num_destination_registers;
}

bool traced_instruction::has_destination_registers() {
    return m_num_destination_registers > 0;
}

std::string traced_instruction::get_pc_string() const {
    return m_pc_string;
}

void traced_instruction::set_pc_string(std::string pc_string) {
    m_pc_string = pc_string;
}

control_bits &traced_instruction::get_control_bits() {
    return *m_control_bits;
}

bool traced_instruction::is_first_operand_of_mref_cbank_desc_using_regular_reg(unsigned int idx_op)
{
    if (idx_op >= m_operands.size()) {
        throw std::out_of_range("traced operand index out of range");
    }
    bool is_first_operand_of_mref_using_regular_reg = false;
    if ((m_operands[idx_op]->get_operand_type() == TraceEnhancedOperandType::MREF) || (m_operands[idx_op]->get_operand_type() == TraceEnhancedOperandType::CBANK) || (m_operands[idx_op]->get_operand_type() == TraceEnhancedOperandType::DESC))
    {
        if ((m_operands[idx_op]->get_operand_type() == TraceEnhancedOperandType::DESC))
        {
            is_first_operand_of_mref_using_regular_reg = true;
        }
        else
        {
            traced_operand &mref = *m_operands[idx_op];
            is_first_operand_of_mref_using_regular_reg = (mref.get_operand_string().find(std::string("R")) != std::string::npos) && (mref.get_operand_string().find(std::string("UR")) == std::string::npos);
        }
    }else {
        bool found = false;
        for(std::size_t i = 0; (i < m_operands.size()) && !found; i++) {
            if((m_operands[i]->get_operand_type() == TraceEnhancedOperandType::MREF)) {
                traced_operand &mref = *m_operands[i];
                is_first_operand_of_mref_using_regular_reg = (mref.get_operand_string().find(std::string("R")) != std::string::npos) && (mref.get_operand_string().find(std::string("UR")) == std::string::npos);
                found = true;
            }else if(m_operands[i]->get_operand_type() == TraceEnhancedOperandType::DESC) {
                is_first_operand_of_mref_using_regular_reg = true;
                found = true;
            }
        }
    }
    return is_first_operand_of_mref_using_regular_reg;
}

Tensor_core_instruction_information& traced_instruction::get_tensor_core_instruction_info() {
    return m_tensor_core_instruction_info;
}

bool traced_instruction::set_tensor_core_instruction_info() {
  if (!is_tensor_core_op()) {
    return false;
  }
  if(!m_tensor_core_instruction_info.is_set) {
    try {
    std::smatch matches;
    std::string op_code = get_op_code();
    std::regex pattern(R"(\d+)");
    std::vector<std::string> numbers;
    bool is_sparse = op_code.find("SP") != std::string::npos;
    bool is_16816_fp32_1688_fp32 = op_code == ("HMMA.16816.F32") || (op_code == "HMMA.1688.F32");
    std::sregex_iterator begin(op_code.begin(), op_code.end(), pattern), end;
    for (std::sregex_iterator i = begin; i != end; ++i) {
        std::smatch match = *i;
        numbers.push_back(match.str());
    }
    if (numbers.empty()) {
      return false;
    }
    unsigned int operand_bit_size = 1;
    if(numbers.size() > 1) {
      operand_bit_size = std::stoi(numbers[1]);
    }
    unsigned int size_m = 0;
    unsigned int size_n = 8;
    unsigned int size_k = 0;
    unsigned int pos_n = 1;
    if(numbers[0].substr(0, 1) == "8") {
      size_m = 8;
    }else if(numbers[0].substr(0, 2) == "16") {
      size_m = 16;
      pos_n = 2;
    }else{
      return false;
    }
    if (pos_n >= numbers[0].size() || numbers[0][pos_n] != '8') {
      return false;
    }
    size_k = std::stoi(numbers[0].substr(pos_n + 1));
    m_tensor_core_instruction_info = Tensor_core_instruction_information(size_m, size_n, size_k, operand_bit_size, is_sparse, is_16816_fp32_1688_fp32);
    } catch (const std::exception &) {
      return false;
    }
  }
  return m_tensor_core_instruction_info.is_set;
}

void traced_instruction::set_simulation_opcode(const std::unordered_map<std::string, OpcodeChar> *OpcodeMap, std::string str_op) {
    std::vector<std::string> opcode_tokens = get_opcode_tokens(str_op);
    if (OpcodeMap == nullptr || opcode_tokens.empty()) {
        m_simulation_opcode = NO_OP;
        return;
    }
    std::string opcode1 = opcode_tokens[0];

    std::unordered_map<std::string, OpcodeChar>::const_iterator it =
    OpcodeMap->find(opcode1);
    if (it != OpcodeMap->end()) {
        m_simulation_opcode = (op_type)(it->second.opcode_category);
    }
}

op_type traced_instruction::get_simulation_opcode() {
    return m_simulation_opcode;
}

bool traced_instruction::is_dp_op() {
    return m_simulation_opcode == DP_OP;
}

bool traced_instruction::is_load_op() {
    return (m_simulation_opcode == LOAD_OP) || (m_simulation_opcode == TENSOR_CORE_LOAD_OP);
}

bool traced_instruction::is_store_op() {
    return (m_simulation_opcode == STORE_OP) || (m_simulation_opcode == TENSOR_CORE_STORE_OP);
}

bool traced_instruction::is_tensor_core_op() {
    return m_simulation_opcode == TENSOR_CORE_OP;
}

bool traced_instruction::Serialize(rapidjson::Writer<rapidjson::StringBuffer> * writer) const {
    writer->StartObject();

    writer->String("pc_string_hex"); 
    writer->String(m_pc_string.c_str());
    writer->String("pc_num_dec"); 
    writer->Int64(m_pc_num);

    writer->String("op_code"); 
    writer->String(m_op_code.c_str());

    writer->String("is_predicated");
    writer->Bool(m_is_predicated);
    writer->String("is_uniform_predicate");
    writer->Bool(m_is_uniform_predicate);
    writer->String("is_predicate_negate");
    writer->Bool(m_is_predicate_negate);
    writer->String("predicate_register");
    writer->Int64(m_predicate_register);

    writer->String("num_destination_registers");
    writer->Int64(m_num_destination_registers);

    writer->String("encoded_instruction");
    writer->StartArray();
    for(auto it = m_encoded_instruction.begin(); it != m_encoded_instruction.end(); it++) {
        writer->String(it->c_str());
    }
    writer->EndArray();
    
    writer->String("operands");
    writer->StartArray();
    for(auto it = m_operands.begin(); it != m_operands.end(); it++) {
        (*it)->Serialize(writer);
    }
    writer->EndArray();

    writer->String("control_bits");
    m_control_bits->Serialize(writer);

    writer->String("register_usage");
    m_register_usage->Serialize(writer);

    writer->EndObject();
    
    return true;
}

void traced_instruction::swap_state(traced_instruction &other) noexcept
{
    using std::swap;
    swap(m_pc_string, other.m_pc_string);
    swap(m_pc_num, other.m_pc_num);
    swap(m_op_code, other.m_op_code);
    swap(m_is_predicated, other.m_is_predicated);
    swap(m_is_uniform_predicate, other.m_is_uniform_predicate);
    swap(m_is_predicate_negate, other.m_is_predicate_negate);
    swap(m_predicate_register, other.m_predicate_register);
    swap(m_encoded_instruction, other.m_encoded_instruction);
    swap(m_operands, other.m_operands);
    swap(m_num_destination_registers, other.m_num_destination_registers);
    swap(m_register_usage, other.m_register_usage);
    swap(m_control_bits, other.m_control_bits);
    swap(m_is_imad, other.m_is_imad);
    swap(m_is_imad_wide, other.m_is_imad_wide);
    swap(m_contains_setp, other.m_contains_setp);
    swap(m_membar_type, other.m_membar_type);
    swap(m_tensor_core_instruction_info, other.m_tensor_core_instruction_info);
    swap(m_simulation_opcode, other.m_simulation_opcode);
}

bool traced_instruction::Deserialize(const rapidjson::Value &obj)
{
    // Decode into an initialized candidate so a malformed nested operand,
    // control block, or register-usage object cannot leave a reused
    // instruction partially updated.
    traced_instruction candidate(
        "", 0, "NOP", std::vector<std::string>{}, false, false, false, 0);
    if (!candidate.deserialize_in_place(obj)) {
        return false;
    }
    swap_state(candidate);
    return true;
}

bool traced_instruction::deserialize_in_place(const rapidjson::Value &obj)
{
    std::string pc_string;
    std::string op_code;
    const auto is_uint_field = [&obj](const char *name) {
        return obj.HasMember(name) && obj[name].IsInt64() &&
               obj[name].GetInt64() >= 0 &&
               static_cast<uint64_t>(obj[name].GetInt64()) <=
                   std::numeric_limits<unsigned int>::max();
    };
    if (!obj.IsObject() || !HasUniqueObjectMembers(obj) ||
        !obj.HasMember("pc_string_hex") ||
        !ReadString(obj["pc_string_hex"], pc_string) ||
        !is_uint_field("pc_num_dec") || !obj.HasMember("op_code") ||
        !ReadString(obj["op_code"], op_code) ||
        get_opcode_tokens(op_code).empty() ||
        !obj.HasMember("is_predicated") ||
        !obj["is_predicated"].IsBool() ||
        !obj.HasMember("is_uniform_predicate") ||
        !obj["is_uniform_predicate"].IsBool() ||
        !obj.HasMember("is_predicate_negate") ||
        !obj["is_predicate_negate"].IsBool() ||
        !is_uint_field("predicate_register") ||
        !is_uint_field("num_destination_registers") ||
        !obj.HasMember("encoded_instruction") ||
        !obj["encoded_instruction"].IsArray() ||
        obj["encoded_instruction"].Size() > 2 ||
        !obj.HasMember("operands") || !obj["operands"].IsArray() ||
        !obj.HasMember("control_bits") || !obj["control_bits"].IsObject() ||
        !obj.HasMember("register_usage") ||
        !obj["register_usage"].IsObject()) {
        return false;
    }

    m_encoded_instruction.clear();
    m_operands.clear();
    m_register_usage.reset();
    m_control_bits.reset();

    m_is_imad = false;
    m_is_imad_wide = false;
    m_membar_type = MEMBAR_TRACED_NONE;
    m_contains_setp = false;
    m_pc_string = std::move(pc_string);
    m_pc_num = obj["pc_num_dec"].GetInt64();
    try {
        std::size_t parsed = 0;
        const unsigned long pc_from_hex =
            std::stoul(m_pc_string, &parsed, 16);
        if (parsed != m_pc_string.size() ||
            pc_from_hex != static_cast<unsigned long>(m_pc_num)) {
            return false;
        }
    } catch (const std::exception &) {
        return false;
    }
    set_op_code(op_code);
    m_simulation_opcode = NO_OP;
    m_is_predicated = obj["is_predicated"].GetBool();
    m_is_uniform_predicate = obj["is_uniform_predicate"].GetBool();
    m_is_predicate_negate = obj["is_predicate_negate"].GetBool();
    m_predicate_register = obj["predicate_register"].GetInt64();
    if ((!m_is_predicated &&
         (m_is_uniform_predicate || m_is_predicate_negate ||
          m_predicate_register != 0)) ||
        (m_is_predicated &&
         m_predicate_register > static_cast<unsigned int>(PT))) {
        return false;
    }

    const rapidjson::Value &encoded_instruction = obj["encoded_instruction"];
    for (rapidjson::SizeType i = 0; i < encoded_instruction.Size(); i++)
    {
        std::string encoded_word;
        if (!ReadString(encoded_instruction[i], encoded_word)) {
            return false;
        }
        m_encoded_instruction.push_back(std::move(encoded_word));
    }

    m_num_destination_registers = obj["num_destination_registers"].GetInt64();
    m_register_usage = std::make_unique<register_usage>();
    if (!m_register_usage->Deserialize(obj["register_usage"])) {
        return false;
    }

    m_control_bits = std::make_unique<control_bits>();
    if (!m_control_bits->Deserialize(obj["control_bits"])) {
        return false;
    }

    const rapidjson::Value &operands = obj["operands"];
    for (rapidjson::SizeType i = 0; i < operands.Size(); i++)
    {
        std::unique_ptr<traced_operand> operand = std::make_unique<traced_operand>();
        if (!operand->Deserialize(operands[i])) {
            return false;
        }
        m_operands.push_back(std::move(operand));
    }

    return true;
}
