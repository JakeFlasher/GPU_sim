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

#include "traced_execution.h"

#include <stdexcept>
#include "string_utilities.h"


traced_execution::traced_execution(std::string benchmark_name) {
    m_benchmark_name = benchmark_name;
}

traced_execution::traced_execution() {
    // Used for Deserializing
}

traced_execution::~traced_execution() {
    for(auto it = m_kernels.begin(); it != m_kernels.end(); it++) {
        delete it->second;
    }
    m_kernels.clear();
}

bool traced_execution::has_kernel(std::string kernel_name) {
    return m_kernels.find(kernel_name) != m_kernels.end();
}

bool traced_execution::has_kernel_with_unique_function_id(unsigned int unique_function_id) {
    return m_function_id_to_kernel_name.find(unique_function_id) != m_function_id_to_kernel_name.end();
}

traced_kernel& traced_execution::get_kernel(std::string kernel_name) {
    return *m_kernels.at(kernel_name);
}

traced_kernel &traced_execution::get_kernel_by_unique_function_id(unsigned int unique_function_id) {
    auto it_funct = m_function_id_to_kernel_name.find(unique_function_id);
    if (it_funct == m_function_id_to_kernel_name.end()) {
        throw std::out_of_range("unknown kernel function id");
    }
    return get_kernel(it_funct->second);
}

unsigned int traced_execution::get_unique_function_id(std::string kernel_name) {
    return m_kernels.at(kernel_name)->get_unique_function_id();
}

search_func_addr_result traced_execution::search_function_addr(uint64_t func_addr) {
    auto it_func = m_func_addr_to_unique_function_id.find(func_addr);
    if(it_func != m_func_addr_to_unique_function_id.end()) {
        auto it_kernel = m_function_id_to_kernel_name.find(it_func->second);
        if (it_kernel == m_function_id_to_kernel_name.end()) {
            return search_func_addr_result();
        }
        std::string kernel_name = it_kernel->second;
        return search_func_addr_result(func_addr, it_func->second, kernel_name, true);
    } else {
        return search_func_addr_result();
    }
}

bool traced_execution::add_traced_kernel(std::string kernel_name, unsigned int unique_function_id, traced_kernel* kernel, uint64_t func_addr, bool is_captured_from_binary) {
    if (kernel == nullptr) {
        return false;
    }
    auto existing = m_kernels.find(kernel_name);
    if (existing != m_kernels.end()) {
        const bool same_identity =
            existing->second->get_unique_function_id() == unique_function_id &&
            existing->second->get_function_addr() == func_addr;
        delete kernel;
        return same_identity;
    }
    m_kernels[kernel_name] = kernel;
    m_kernels[kernel_name]->set_unique_function_id(unique_function_id);
    m_kernels[kernel_name]->set_function_addr(func_addr);
    m_kernels[kernel_name]->set_is_captured_from_binary(is_captured_from_binary);
    const bool maps_valid = generate_kernel_maps();
    if (!maps_valid) {
        auto inserted = m_kernels.find(kernel_name);
        if (inserted != m_kernels.end()) {
            delete inserted->second;
            m_kernels.erase(inserted);
        }
        (void)generate_kernel_maps();
        return false;
    }
    return maps_valid;
}


bool traced_execution::add_no_binary_kernel(std::string kernel_name, unsigned int unique_function_id, uint64_t func_addr, unsigned int architecture_version, std::map<int,std::string> &captured_instrs , bool is_captured_from_binary) {
    auto existing = m_kernels.find(kernel_name);
    if (existing != m_kernels.end()) {
        // Idempotent re-registration is allowed when the incoming
        // (kernel_name, unique_function_id, func_addr) tuple matches
        // the existing entry. NVBit's enhanced-tracer atexit hook can
        // observe the same kernel through more than one ELF extraction
        // path (e.g. multiple cubin sections for a single launched
        // function); a duplicate-but-identical registration must not
        // abort the host process. True conflicts (same kernel name
        // bound to a different unique_function_id or func_addr) return false
        // so the caller can handle the conflict in every build.
        traced_kernel *prior = existing->second;
        bool same_identity =
            prior->get_unique_function_id() == unique_function_id &&
            prior->get_function_addr() == func_addr;
        return same_identity;
    }
    traced_kernel *kernel = new traced_kernel(kernel_name, architecture_version);
    for(auto inst : captured_instrs) {
        if (inst.first < 0) {
            delete kernel;
            return false;
        }
        if (!kernel->add_no_binary_instruction(
                inst.first, ReplaceAll(inst.second, ";", " "))) {
            delete kernel;
            return false;
        }
    }
    m_kernels[kernel_name] = kernel;
    m_kernels[kernel_name]->set_unique_function_id(unique_function_id);
    m_kernels[kernel_name]->set_function_addr(func_addr);
    m_kernels[kernel_name]->set_is_captured_from_binary(is_captured_from_binary);
    const bool maps_valid = generate_kernel_maps();
    if (!maps_valid) {
        auto inserted = m_kernels.find(kernel_name);
        if (inserted != m_kernels.end()) {
            delete inserted->second;
            m_kernels.erase(inserted);
        }
        (void)generate_kernel_maps();
        return false;
    }
    return maps_valid;
}

bool traced_execution::add_instruction_to_a_kernel(std::string kernel_name, std::vector<std::string> instruction_part1, std::vector<std::string> instruction_part2, int threshold_unique_kernel_checking, std::string full_instruction_str) {
    auto it = m_kernels.find(kernel_name);
    if (it == m_kernels.end() || it->second == nullptr) {
        return false;
    }
    return it->second->add_instruction(
        instruction_part1, instruction_part2, threshold_unique_kernel_checking,
        full_instruction_str);
}

bool traced_execution::add_register_usage_to_a_kernel_instruction(std::string kernel_name, unsigned pc, std::string reg_file_name, unsigned num_regs) {
    auto it = m_kernels.find(kernel_name);
    if (it == m_kernels.end()) {
        return false;
    }
    return it->second->add_register_usage(pc, reg_file_name, num_regs);
}

bool traced_execution::add_register_usage_to_a_kernel(std::string kernel_name, std::map<unsigned int, std::map<std::string, unsigned int>> &reg_usage, std::map<unsigned int, std::tuple<std::string, int>> &call_target_by_pc) {
    if (m_kernels.find(kernel_name) == m_kernels.end()) {
        return false;
    }
    for(auto inst_entry : reg_usage) {
        for(auto reg_file : inst_entry.second) {
            if (!add_register_usage_to_a_kernel_instruction(
                    kernel_name, inst_entry.first, reg_file.first,
                    reg_file.second)) {
                return false;
            }
        }
    }
    for(auto call_target : call_target_by_pc) {
        auto it = m_kernels.find(kernel_name);
        if (it == m_kernels.end() || it->second == nullptr ||
            !it->second->add_call_target(
                call_target.first, std::get<0>(call_target.second),
                std::get<1>(call_target.second))) {
            return false;
        }
    }
    return true;
}

std::string traced_execution::get_benchmark_name() const {
    return m_benchmark_name;
}

void traced_execution::set_benchmark_name(std::string benchmark_name) {
    m_benchmark_name = benchmark_name;
}

void traced_execution::remove_useless_kernels() {
    for (auto it = m_kernels.begin(); it != m_kernels.end();) {
        if(!it->second->is_kernel_in_use()) {
            delete it->second;
            it = m_kernels.erase(it);
        } else {
            ++it;
        }
    }
    (void)generate_kernel_maps();
}

bool traced_execution::remove_specific_kernel(std::string kernel_name) {
    auto it = m_kernels.find(kernel_name);
    if (it == m_kernels.end()) {
        return false;
    }
    delete it->second;
    m_kernels.erase(it);
    (void)generate_kernel_maps();
    return true;
}

bool traced_execution::Serialize(rapidjson::Writer<rapidjson::StringBuffer> * writer) const {
    writer->StartObject();

    writer->String("benchmark_name"); 
    writer->String(m_benchmark_name.c_str());
    
    writer->String("kernels");
    writer->StartArray();
    for(auto it = m_kernels.begin(); it != m_kernels.end(); it++) {
        it->second->Serialize(writer);
    }
    writer->EndArray();

    writer->EndObject();
    
    return true;
}

bool traced_execution::generate_kernel_maps() {
    std::map<unsigned int, std::string> function_id_map;
    std::map<uint64_t, unsigned int> function_address_map;
    for(auto kernel : m_kernels) {
        unsigned int function_id = kernel.second->get_unique_function_id();
        uint64_t kernel_func_addr = kernel.second->get_function_addr();
        auto function_inserted = function_id_map.emplace(function_id, kernel.first);
        if (!function_inserted.second) {
            m_function_id_to_kernel_name.clear();
            m_func_addr_to_unique_function_id.clear();
            return false;
        }
        auto address_inserted =
            function_address_map.emplace(kernel_func_addr, function_id);
        if (!address_inserted.second) {
            m_function_id_to_kernel_name.clear();
            m_func_addr_to_unique_function_id.clear();
            return false;
        }
    }
    m_function_id_to_kernel_name.swap(function_id_map);
    m_func_addr_to_unique_function_id.swap(function_address_map);
    return true;
}

bool traced_execution::Deserialize(const rapidjson::Value &obj)
{
    std::string benchmark_name;
    if (!obj.IsObject() || !HasUniqueObjectMembers(obj) ||
        !obj.HasMember("benchmark_name") ||
        !ReadString(obj["benchmark_name"], benchmark_name) || !obj.HasMember("kernels") ||
        !obj["kernels"].IsArray()) {
        return false;
    }

    std::map<std::string, traced_kernel *> parsed_kernels;
    const auto discard_kernels = [](std::map<std::string, traced_kernel *> &kernels) {
        for (auto &entry : kernels) {
            delete entry.second;
        }
        kernels.clear();
    };
    const rapidjson::Value &kernels = obj["kernels"];
    for (rapidjson::SizeType i = 0; i < kernels.Size(); i++)
    {
        traced_kernel *kernel = new traced_kernel();
        if (!kernel->Deserialize(kernels[i]) ||
            parsed_kernels.find(kernel->get_kernel_name()) != parsed_kernels.end()) {
            delete kernel;
            discard_kernels(parsed_kernels);
            return false;
        }
        parsed_kernels[kernel->get_kernel_name()] = kernel;
    }

    // Publish the candidate only after every nested kernel has decoded.  Keep
    // the old object alive until map generation succeeds so a failed retry
    // cannot erase a previously usable execution description.
    auto previous_kernels = std::move(m_kernels);
    auto previous_function_ids = std::move(m_function_id_to_kernel_name);
    auto previous_function_addresses = std::move(m_func_addr_to_unique_function_id);
    const std::string previous_benchmark_name = std::move(m_benchmark_name);
    m_kernels = std::move(parsed_kernels);
    m_function_id_to_kernel_name.clear();
    m_func_addr_to_unique_function_id.clear();
    m_benchmark_name = benchmark_name;

    if (!generate_kernel_maps()) {
        discard_kernels(m_kernels);
        m_kernels = std::move(previous_kernels);
        m_function_id_to_kernel_name = std::move(previous_function_ids);
        m_func_addr_to_unique_function_id = std::move(previous_function_addresses);
        m_benchmark_name = previous_benchmark_name;
        return false;
    }

    for (auto &entry : previous_kernels) {
        delete entry.second;
    }
    return true;
}

void traced_execution::print(FILE *fp) {
    fprintf(fp, "Benchmark: %s\n", m_benchmark_name.c_str());
    for(auto it = m_kernels.begin(); it != m_kernels.end(); it++) {
        fprintf(fp, "Kernel: %s\n", it->first.c_str());
    }
}
