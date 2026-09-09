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

#include <string>
#include "JSONBase.h"
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <utility>

namespace {

constexpr std::uint64_t kMaxEnhancedJsonBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaxEnhancedJsonDepth = 1024;

bool HasUniqueMembers(const rapidjson::Value &value, std::size_t depth = 0) {
	if (depth > kMaxEnhancedJsonDepth) {
		return false;
	}
	if (value.IsString()) {
		for (rapidjson::SizeType i = 0; i < value.GetStringLength(); ++i) {
			if (value.GetString()[i] == '\0') {
				return false;
			}
		}
	} else if (value.IsObject()) {
		std::set<std::string> members;
		for (auto it = value.MemberBegin(); it != value.MemberEnd(); ++it) {
			if (!HasUniqueMembers(it->name, depth + 1)) {
				return false;
			}
			const rapidjson::SizeType name_length = it->name.GetStringLength();
			const char *name_data = it->name.GetString();
			if (name_length != 0 && name_data == nullptr) {
				return false;
			}
			const std::string name =
				name_length == 0 ? std::string() : std::string(name_data, name_length);
			if (!members.emplace(name).second ||
				!HasUniqueMembers(it->value, depth + 1)) {
				return false;
			}
		}
	} else if (value.IsArray()) {
		for (auto it = value.Begin(); it != value.End(); ++it) {
			if (!HasUniqueMembers(*it, depth + 1)) {
				return false;
			}
		}
	}
	return true;
}

} // namespace

bool JSONBase::ReadString(const rapidjson::Value &value, std::string &out)
{
	if (!value.IsString())
		return false;
	const rapidjson::SizeType length = value.GetStringLength();
	const char *data = value.GetString();
	if (length != 0 && data == nullptr)
		return false;
	for (rapidjson::SizeType i = 0; i < length; ++i) {
		if (data[i] == '\0')
			return false;
	}
	if (length == 0) {
		out.clear();
		return true;
	}
	out.assign(data, length);
	return true;
}

bool JSONBase::HasUniqueObjectMembers(const rapidjson::Value &value)
{
	if (!value.IsObject())
		return false;
	std::set<std::string> members;
	for (auto it = value.MemberBegin(); it != value.MemberEnd(); ++it) {
		std::string name;
		if (!ReadString(it->name, name) || !members.emplace(std::move(name)).second)
			return false;
	}
	return true;
}


std::string JSONBase::Serialize() const
{
	rapidjson::StringBuffer ss;
	rapidjson::Writer<rapidjson::StringBuffer> writer(ss);		// Can also use Writer for condensed formatting
	if (Serialize(&writer))
		return ss.GetString();
	return "";
}

bool JSONBase::Deserialize(const std::string& s)
{
	rapidjson::Document doc;
	if (!InitDocument(s, doc))
		return false;

	return Deserialize(doc);
}

bool JSONBase::DeserializeFromFile(const std::string& filePath)
{
	std::ifstream f(filePath, std::ios::in | std::ios::binary);
	if (!f) return false;
	f.seekg(0, std::ios::end);
	const std::streamoff size = f.tellg();
	if (size < 0 || static_cast<std::uint64_t>(size) > kMaxEnhancedJsonBytes ||
	    static_cast<std::uint64_t>(size) >
	        static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
		return false;
	}
	f.seekg(0, std::ios::beg);
	std::string input(static_cast<std::size_t>(size), '\0');
	if (size > 0) {
		f.read(input.data(), static_cast<std::streamsize>(size));
		if (f.bad() || f.gcount() != static_cast<std::streamsize>(size)) {
			return false;
		}
	}
	f.close();
	if (!f) return false;

	return Deserialize(input);
}

bool JSONBase::SerializeToFile(const std::string& filePath)
{
	std::ofstream f(filePath);
	if (!f) return false;
	std::string s = Serialize();
	if (s.empty()) return false;
	f << s;
	if (!f) return false;
	f.flush();
	if (!f) return false;
	f.close();
	return static_cast<bool>(f);
}

bool JSONBase::InitDocument(const std::string& s, rapidjson::Document& doc)
{
	if (s.empty())
		return false;

	// RapidJSON's memory stream treats a raw NUL as end-of-input even when a
	// length is supplied. Reject it explicitly so trailing bytes cannot be
	// hidden behind a zero byte.
	if (s.find('\0') != std::string::npos) {
		return false;
	}

	// Use the explicit byte length: the zero-terminated overload silently
	// ignores any bytes after an embedded NUL in a std::string.
	if (doc.Parse<rapidjson::kParseIterativeFlag |
	             rapidjson::kParseValidateEncodingFlag>(s.data(), s.size())
	            .HasParseError()) {
		return false;
	}
	return HasUniqueMembers(doc);
}
