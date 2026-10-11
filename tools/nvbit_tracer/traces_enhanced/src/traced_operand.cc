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

#include "traced_operand.h"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "string_utilities.h"

namespace {

bool has_empty_plus_component(const std::string &text) {
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '+' &&
            (i == 0 || i + 1 == text.size() || text[i + 1] == '+')) {
            return true;
        }
    }
    return false;
}

} // namespace

traced_operand::traced_operand(std::string operand_string)
{
    if (operand_string.empty()) {
        throw std::invalid_argument("empty operand");
    }
    m_is_reuse_bit_set = false;
    m_has_inmediate = false;
    m_has_reg = false;

    bool is_other_wait_barriers_identifiers = false;

    std::vector<std::string> splitted_operand_string = split_string(operand_string, '.');

    for (unsigned i = 1; i < splitted_operand_string.size(); i++)
    {
        std::string current_modifier = splitted_operand_string[i];
        std::size_t plus_pos = current_modifier.find("+");
        if (std::string::npos != plus_pos)
        {
            current_modifier = current_modifier.substr(0, plus_pos);
        }
        if (current_modifier.find("reuse") != std::string::npos)
        {
            m_is_reuse_bit_set = true;
        }
        current_modifier = ReplaceAll(current_modifier, "]", "");
        m_modifiers.push_back(current_modifier);
    }

    m_operand_string = operand_string;

    unsigned idx_first_reg_num = 0;

    m_is_negative_reg = false;
    m_is_absolute_reg = false;
    if (operand_string.size() >= 2 && (operand_string[0] == '-') && ((operand_string[1] == 'R') || (operand_string[1] == 'U') || (operand_string[1] == 'P') || (operand_string[1] == 'c')))
    {
        idx_first_reg_num++;
        m_is_negative_reg = true;
    }
    else if (operand_string.size() >= 2 && (operand_string[0] == '|') && ((operand_string[1] == 'R') || (operand_string[1] == 'U') || (operand_string[1] == 'P') || (operand_string[1] == 'c')))
    {
        idx_first_reg_num++;
        m_is_absolute_reg = true;
    }else if (operand_string.size() >= 2 && (operand_string[0] == '~') && ((operand_string[1] == 'R') || (operand_string[1] == 'U') || (operand_string[1] == 'P') || (operand_string[1] == 'c')))
    {
        idx_first_reg_num++;
    }else if( operand_string.size() >= 3 && (operand_string[0] == '-') && (operand_string[1] == '|') && ((operand_string[2] == 'R') || (operand_string[2] == 'U') || (operand_string[2] == 'P') || (operand_string[1] == 'c'))){
        idx_first_reg_num += 2;
        m_is_negative_reg = true;
        m_is_absolute_reg = true;
    }

    bool skip_idx_inc = false;

    if (idx_first_reg_num >= operand_string.size()) {
        throw std::invalid_argument("operand prefix has no payload");
    }

    // Rpc is a control-flow register, not general register R0. Preserve the
    // exact spelling/modifier without inventing a numeric GPR identity.
    if (operand_string.compare(idx_first_reg_num, 4, "Rpc.") == 0)
    {
        m_operand_type = TraceEnhancedOperandType::GENERIC;
        skip_idx_inc = true;
    }
    else if (operand_string[idx_first_reg_num] == 'R')
    {
        m_operand_type = TraceEnhancedOperandType::REG;
    }
    else if ((operand_string[idx_first_reg_num] == 'P') || (operand_string[idx_first_reg_num] == '!') || (operand_string.find("UP") != std::string::npos))
    {
        if (operand_string[idx_first_reg_num] == '!')
        {
            idx_first_reg_num++;
            m_is_negative_reg = true;
        }
        if (operand_string.find("UP") != std::string::npos)
        {
            m_operand_type = TraceEnhancedOperandType::UPRED;
            idx_first_reg_num++;
        }
        else
        {
            m_operand_type = TraceEnhancedOperandType::PRED;
        }
    }
    else if (operand_string[idx_first_reg_num] == 'B')
    {
        m_operand_type = TraceEnhancedOperandType::BREG;
    }
    else if (operand_string[idx_first_reg_num] == 'c')
    {
        m_operand_type = TraceEnhancedOperandType::CBANK;
    }
    else if (operand_string[idx_first_reg_num] == '[')
    {
        m_operand_type = TraceEnhancedOperandType::MREF;
    }
    else if (operand_string.find("SR") != std::string::npos)
    {
        m_operand_type = TraceEnhancedOperandType::SR;
        idx_first_reg_num++;
    }
    else if (operand_string.find("SB") != std::string::npos)
    {
        m_operand_type = TraceEnhancedOperandType::SB;
        idx_first_reg_num++;
    }else if(operand_string.find("{") != std::string::npos) {
        m_operand_type = TraceEnhancedOperandType::SB;
        is_other_wait_barriers_identifiers = true;
    }
    else if (operand_string.find("desc") != std::string::npos)
    {
        m_operand_type = TraceEnhancedOperandType::DESC;
        idx_first_reg_num += 3;
    }
    else if (operand_string.find("UR") != std::string::npos)
    {
        m_operand_type = TraceEnhancedOperandType::UREG;
        idx_first_reg_num++;
    }
    else
    {
        skip_idx_inc = true;
        if (operand_string.find("0x") == std::string::npos &&
            operand_string.find("0X") == std::string::npos)
        {
            try
            {
                if (operand_string.find("QNAN") != std::string::npos)
                {
                    operand_string = ReplaceAll(operand_string, "QNAN", "NAN");
                }
                std::size_t parsed = 0;
                std::stod(operand_string, &parsed);
                if (parsed != operand_string.size()) {
                    throw std::invalid_argument(
                        "immediate operand has trailing characters");
                }
                m_operand_type = TraceEnhancedOperandType::IMM_DOUBLE;
            }
            catch (const std::invalid_argument &)
            {
                throw;
            }
        }
        else
        {
            m_operand_type = TraceEnhancedOperandType::IMM_UINT64;
        }
    }
    if (!skip_idx_inc)
    {
        idx_first_reg_num++;
    }

    // std::string number_string = operand_string.substr(idx_first_reg_num, operand_string.size() - idx_first_reg_num);
    std::string number_string = operand_string.substr(idx_first_reg_num);
    number_string = ReplaceAll(number_string, "|", "");
    number_string = ReplaceAll(number_string, "~", "");

    m_operand_reg_number = 0;
    switch (m_operand_type)
    {
    case TraceEnhancedOperandType::SB:
        if(is_other_wait_barriers_identifiers) {
            number_string = ReplaceAll(number_string, "{", "");
            number_string = ReplaceAll(number_string, "}", "");
            // Blackwell emits DEPBAR wait sets as either `{5 4 3 2 1 0}`
            // or `{5,4,3,2,1,0}` depending on the disassembler surface.
            // The old comma-only split fed the complete space-separated
            // string to stod/stoull and rejected an otherwise valid control
            // instruction, forcing tracer_tool into a NO_REGS/raw-SASS
            // fallback.  Normalize both separators while retaining every
            // original immediate as an operand value.
            for (char &character : number_string) {
                if (character == ',') character = ' ';
            }
            std::istringstream values(number_string);
            std::string value;
            while (values >> value) {
                add_inmediate(value, TraceEnhancedOperandType::IMM_DOUBLE);
            }
            if (m_operands_inmediates.empty()) {
                throw std::invalid_argument("empty wait-barrier identifier set");
            }
        }else{
            set_reg_num(number_string, 99999, '-');
        }
        break;
    case TraceEnhancedOperandType::BREG:
        set_reg_num(number_string, 99999, '-');
        break;
    case TraceEnhancedOperandType::UREG:
        set_reg_num(number_string, URZ, 'Z');
        break;
    case TraceEnhancedOperandType::REG:
        set_reg_num(number_string, RZ, 'Z');
        break;
    case TraceEnhancedOperandType::UPRED:
        set_reg_num(number_string, UPT, 'T');
        break;
    case TraceEnhancedOperandType::PRED:
        set_reg_num(number_string, PT, 'T');
        break;
    case TraceEnhancedOperandType::IMM_UINT64:
        add_inmediate(number_string, TraceEnhancedOperandType::IMM_UINT64);
        break;
    case TraceEnhancedOperandType::IMM_DOUBLE:
        add_inmediate(number_string, TraceEnhancedOperandType::IMM_DOUBLE);
        break;
    case TraceEnhancedOperandType::MREF:
    {
        if (number_string.size() < 2 || number_string.back() != ']' ||
            std::count(number_string.begin(), number_string.end(), '[') != 0 ||
            std::count(number_string.begin(), number_string.end(), ']') != 1) {
            throw std::invalid_argument("malformed memory reference operand");
        }
        const std::string memory_reference =
            number_string.substr(0, number_string.size() - 1);
        if (memory_reference.empty() ||
            has_empty_plus_component(memory_reference)) {
            throw std::invalid_argument("malformed memory reference operand");
        }
        std::string cleaned_operands = ReplaceAll(number_string, "]", "");
        std::vector<std::string> splitted_operands = split_string(cleaned_operands, '+');
        for (unsigned i = 0; i < splitted_operands.size(); i++)
        {
            if ((splitted_operands[i][0] == 'R') || (splitted_operands[i][0] == 'U'))
            {
                int reserved_reg = RZ;
                int first_substring_pos = 1;
                const std::size_t search_dot = splitted_operands[i].find_first_of('.');
                const std::size_t register_end =
                    search_dot == std::string::npos ? splitted_operands[i].size()
                                                    : search_dot;
                if (splitted_operands[i][0] == 'U')
                {
                    reserved_reg = URZ;
                    first_substring_pos++;
                }
                if (register_end <= static_cast<std::size_t>(first_substring_pos)) {
                    throw std::invalid_argument("empty memory-reference register");
                }
                std::string reg_string = splitted_operands[i].substr(
                    first_substring_pos,
                    register_end - static_cast<std::size_t>(first_substring_pos));
                set_reg_num(reg_string, reserved_reg, 'Z');
            }
            else
            {
                add_inmediate(splitted_operands[i], TraceEnhancedOperandType::IMM_DOUBLE);
            }
        }
    }
    break;
    case TraceEnhancedOperandType::CBANK:
    {
        if (number_string.size() < 4 || number_string.front() != '[' ||
            number_string.back() != ']' ||
            std::count(number_string.begin(), number_string.end(), '[') != 2 ||
            std::count(number_string.begin(), number_string.end(), ']') != 2) {
            throw std::invalid_argument("malformed constant-bank operand");
        }
        std::string cleaned_operands = ReplaceAll(number_string, "]", "");
        std::vector<std::string> splitted_operands = split_string(cleaned_operands, '[');
        if (splitted_operands.size() != 2 || splitted_operands[0].empty() ||
            splitted_operands[1].empty() ||
            std::any_of(splitted_operands.begin(), splitted_operands.end(),
                        [](const std::string &part) {
                            return has_empty_plus_component(part);
                        })) {
            throw std::invalid_argument("malformed constant-bank operand");
        }
        for (unsigned i = 0; i < splitted_operands.size(); i++)
        {
            std::vector<std::string> splitted_operands2 = split_string(splitted_operands[i], '+');
            for (unsigned j = 0; j < splitted_operands2.size(); j++)
            {
                if ((splitted_operands2[j][0] == 'R') || (splitted_operands2[j][0] == 'U'))
                {
                    int reserved_reg = RZ;
                    int first_substring_pos = 1;
                    const std::size_t search_dot =
                        splitted_operands2[j].find_first_of('.');
                    const std::size_t register_end =
                        search_dot == std::string::npos
                            ? splitted_operands2[j].size()
                            : search_dot;
                    if (splitted_operands2[j][0] == 'U')
                    {
                        reserved_reg = URZ;
                        first_substring_pos++;
                    }
                    if (register_end <= static_cast<std::size_t>(first_substring_pos)) {
                        throw std::invalid_argument("empty constant-bank register");
                    }
                    std::string reg_string = splitted_operands2[j].substr(
                        first_substring_pos,
                        register_end - static_cast<std::size_t>(first_substring_pos));
                    set_reg_num(reg_string, reserved_reg, 'Z');
                }
                else
                {
                    add_inmediate(splitted_operands2[j], TraceEnhancedOperandType::IMM_DOUBLE);
                }
            }
        }
    }
    break;
    default:
        break;
    }
}

traced_operand::traced_operand(std::string operand_string, int target_unique_function_id) {
    m_operand_string = operand_string;
    m_operand_reg_number = target_unique_function_id;
    m_operand_type = TraceEnhancedOperandType::CALL_TARGET;
    m_has_reg = false; // We treated as false because it is not a true Register what we store here
    m_has_inmediate = false;
    m_is_reuse_bit_set = false;
    m_is_negative_reg = false;
    m_is_absolute_reg = false;
    m_modifiers.clear();
    m_operands_inmediates.clear();
}

traced_operand::traced_operand()
    : traced_operand("RZ") {}

void traced_operand::set_reg_num(std::string operand_string, int reserved_reg_num, char reserved_reg_char)
{
    const std::size_t modifier = operand_string.find('.');
    const std::string register_token = operand_string.substr(0, modifier);
    if (register_token.empty()) {
        throw std::invalid_argument("empty register operand");
    }
    if (register_token.front() == '+' || register_token.front() == '-') {
        throw std::invalid_argument("register operand has a sign");
    }
    if (register_token.size() == 1 && register_token[0] == reserved_reg_char)
    {
        m_operand_reg_number = reserved_reg_num;
    }
    else if (reserved_reg_char == 'T' && register_token == "R")
    {
        m_operand_reg_number = PR;
    }
    else
    {
        std::size_t parsed = 0;
        const unsigned long value = std::stoul(register_token, &parsed, 10);
        if (parsed != register_token.size() ||
            (reserved_reg_char == 'T' &&
             value > static_cast<unsigned long>(PR)) ||
            value > static_cast<unsigned long>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("register operand is outside its register-file domain");
        }
        m_operand_reg_number = static_cast<int>(value);
    }
    m_has_reg = true;
}

void traced_operand::add_inmediate(std::string operand_string, TraceEnhancedOperandType operand_type)
{
    double inmediate;
    if (operand_type == TraceEnhancedOperandType::IMM_DOUBLE)
    {
        std::size_t parsed = 0;
        inmediate = std::stod(operand_string, &parsed);
        if (parsed != operand_string.size()) {
            throw std::invalid_argument("immediate operand has trailing characters");
        }
    }
    else
    {
        std::size_t parsed = 0;
        const unsigned long long value =
            std::stoull(operand_string, &parsed, 16);
        if (parsed != operand_string.size()) {
            throw std::invalid_argument("hex immediate operand has trailing characters");
        }
        inmediate = static_cast<double>(value);
    }
    m_operands_inmediates.push_back(inmediate);
    m_has_inmediate = true;
}

TraceEnhancedOperandType traced_operand::get_operand_type() const
{
    return m_operand_type;
}

TraceEnhancedOperandType traced_operand::get_operand_type(std::string operand_string) const
{
    if (operand_string == "IMM_UINT64")
    {
        return TraceEnhancedOperandType::IMM_UINT64;
    }
    else if (operand_string == "IMM_DOUBLE")
    {
        return TraceEnhancedOperandType::IMM_DOUBLE;
    }
    else if (operand_string == "REG")
    {
        return TraceEnhancedOperandType::REG;
    }
    else if (operand_string == "PRED")
    {
        return TraceEnhancedOperandType::PRED;
    }
    else if (operand_string == "UREG")
    {
        return TraceEnhancedOperandType::UREG;
    }
    else if (operand_string == "UPRED")
    {
        return TraceEnhancedOperandType::UPRED;
    }
    else if (operand_string == "CBANK")
    {
        return TraceEnhancedOperandType::CBANK;
    }
    else if (operand_string == "MREF")
    {
        return TraceEnhancedOperandType::MREF;
    }
    else if (operand_string == "GENERIC")
    {
        return TraceEnhancedOperandType::GENERIC;
    }
    else if (operand_string == "BREG")
    {
        return TraceEnhancedOperandType::BREG;
    }
    else if (operand_string == "SR")
    {
        return TraceEnhancedOperandType::SR;
    }
    else if (operand_string == "SB")
    {
        return TraceEnhancedOperandType::SB;
    }
    else if (operand_string == "DESC")
    {
        return TraceEnhancedOperandType::DESC;
    }else if(operand_string == "CALL_TARGET") {
        return TraceEnhancedOperandType::CALL_TARGET;
    }else
    {
        // Handle unknown operand types
        throw std::invalid_argument("Unknown operand type: " + operand_string);
    }
}

std::string traced_operand::get_operand_string() const
{
    return m_operand_string;
}

void traced_operand::set_operand_string(std::string operand_string)
{
    m_operand_string = operand_string;
}

long traced_operand::get_operand_reg_number() const
{
    return m_operand_reg_number;
}

bool traced_operand::get_has_reg() const
{
    return m_has_reg;
}

bool traced_operand::get_is_type_using_only_gathered_registers() const {
    bool res = false;
    if((m_operand_type == TraceEnhancedOperandType::REG) || (m_operand_type == TraceEnhancedOperandType::UREG) ||
        (m_operand_type == TraceEnhancedOperandType::UPRED) || (m_operand_type == TraceEnhancedOperandType::PRED)) {
        res = true;
    }
    return res;
}

bool traced_operand::get_has_inmediate() const
{
    return m_has_inmediate;
}

std::vector<double> traced_operand::get_operands_inmediates() const
{
    return m_operands_inmediates;
}

bool traced_operand::is_reuse_bit_set() const
{
    return m_is_reuse_bit_set;
}

bool traced_operand::Serialize(rapidjson::Writer<rapidjson::StringBuffer> *writer) const
{
    writer->StartObject();

    writer->String("operand_string");
    writer->String(m_operand_string.c_str());

    writer->String("operand_type");
    writer->String(TraceEnhancedOperandTypeStr[static_cast<int>(m_operand_type)]);

    writer->String("has_reg");
    writer->Bool(m_has_reg);

    writer->String("is_negative_reg");
    writer->Bool(m_is_negative_reg);

    writer->String("is_absolute_reg");
    writer->Bool(m_is_absolute_reg);

    writer->String("operand_reg_number");
    writer->Int(m_operand_reg_number);

    writer->String("has_inmediate");
    writer->Bool(m_has_inmediate);

    writer->String("operand_inmmediates");
    writer->StartArray();
    for (unsigned i = 0; i < m_operands_inmediates.size(); i++)
    {
        writer->Double(m_operands_inmediates[i]);
    }
    writer->EndArray();

    writer->String("is_reuse_bit_set");
    writer->Bool(m_is_reuse_bit_set);

    writer->String("operand_modifiers");
    writer->StartArray();
    for (unsigned i = 0; i < m_modifiers.size(); i++)
    {
        writer->String(m_modifiers[i].c_str());
    }
    writer->EndArray();

    writer->EndObject();

    return true;
}

bool traced_operand::Deserialize(const rapidjson::Value &obj)
{
    std::string operand_string;
    std::string operand_type_name;
    if (!obj.IsObject() || !HasUniqueObjectMembers(obj) ||
        !obj.HasMember("operand_string") ||
        !ReadString(obj["operand_string"], operand_string) ||
        !obj.HasMember("operand_modifiers") ||
        !obj["operand_modifiers"].IsArray() ||
        !obj.HasMember("operand_type") ||
        !ReadString(obj["operand_type"], operand_type_name) || !obj.HasMember("has_reg") ||
        !obj["has_reg"].IsBool() || !obj.HasMember("is_negative_reg") ||
        !obj["is_negative_reg"].IsBool() ||
        !obj.HasMember("is_absolute_reg") ||
        !obj["is_absolute_reg"].IsBool() ||
        !obj.HasMember("operand_reg_number") ||
        !obj["operand_reg_number"].IsInt() ||
        !obj.HasMember("has_inmediate") ||
        !obj["has_inmediate"].IsBool() ||
        !obj.HasMember("operand_inmmediates") ||
        !obj["operand_inmmediates"].IsArray() ||
        !obj.HasMember("is_reuse_bit_set") ||
        !obj["is_reuse_bit_set"].IsBool()) {
        return false;
    }
    std::vector<std::string> parsed_modifiers;
    const rapidjson::Value &modifiers = obj["operand_modifiers"];
    for (rapidjson::SizeType i = 0; i < modifiers.Size(); i++)
    {
        std::string modifier;
        if (!ReadString(modifiers[i], modifier)) {
            return false;
        }
        parsed_modifiers.push_back(std::move(modifier));
    }
    TraceEnhancedOperandType operand_type;
    try {
        operand_type = get_operand_type(operand_type_name);
    } catch (const std::exception &) {
        return false;
    }
    const bool has_reg = obj["has_reg"].GetBool();
    const bool is_negative_reg = obj["is_negative_reg"].GetBool();
    const bool is_absolute_reg = obj["is_absolute_reg"].GetBool();
    const int operand_reg_number = obj["operand_reg_number"].GetInt();
    const bool has_inmediate = obj["has_inmediate"].GetBool();

    const int register_limit =
        operand_type == TraceEnhancedOperandType::REG
            ? RZ
        : operand_type == TraceEnhancedOperandType::UREG
                ? URZ_BLACKWELL_PLUS
                : operand_type == TraceEnhancedOperandType::PRED ||
                        operand_type == TraceEnhancedOperandType::UPRED
                    ? PR
                    : -1;
    if (register_limit >= 0 &&
        (operand_reg_number < 0 || operand_reg_number > register_limit)) {
        return false;
    }

    std::vector<double> parsed_inmediates;
    const rapidjson::Value &operands_inmediates = obj["operand_inmmediates"];
    for (rapidjson::SizeType i = 0; i < operands_inmediates.Size(); i++)
    {
        if (!operands_inmediates[i].IsNumber()) {
            return false;
        }
        parsed_inmediates.push_back(operands_inmediates[i].GetDouble());
    }

    const bool is_reuse_bit_set = obj["is_reuse_bit_set"].GetBool();

    // Commit only after every field and nested value has validated.  Callers
    // may reuse an operand object after a malformed retry.
    m_operand_string = operand_string;
    m_modifiers = std::move(parsed_modifiers);
    m_operand_type = operand_type;
    m_has_reg = has_reg;
    m_is_negative_reg = is_negative_reg;
    m_is_absolute_reg = is_absolute_reg;
    m_operand_reg_number = operand_reg_number;
    m_has_inmediate = has_inmediate;
    m_operands_inmediates = std::move(parsed_inmediates);
    m_is_reuse_bit_set = is_reuse_bit_set;

    return true;
}
