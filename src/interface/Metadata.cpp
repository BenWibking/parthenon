//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

// Self Include
#include "Metadata.hpp"

// STL Includes
#include <exception>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using parthenon::MetadataFlag;
using parthenon::Metadata;

namespace parthenon {
// Must declare the flag values for ODR-uses
#define PARTHENON_INTERNAL_FOR_FLAG(name) \
    constexpr MetadataFlag Metadata::name;

    PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG
#undef PARTHENON_INTERNAL_FOR_FLAG

namespace internal {

class UserMetadataState {
 public:
    UserMetadataState() {
#define PARTHENON_INTERNAL_FOR_FLAG(name) \
    flag_name_map_.push_back(#name);

    PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG

#undef PARTHENON_INTERNAL_FOR_FLAG
    }

    MetadataFlag AllocateNewFlag(std::string &&name) {
        if (flag_names_.find(name) != flag_names_.end()) {
            std::stringstream ss;
            ss << "MetadataFlag with name '" << name << "' already exists.";
            throw std::runtime_error(ss.str());
        }

        auto const flag = flag_name_map_.size();
        flag_names_.insert(name);
        flag_name_map_.push_back(std::move(name));
        return MetadataFlag(static_cast<int>(flag));
    }

    std::string const &FlagName(MetadataFlag flag) {
        return flag_name_map_.at(flag.flag_);
    }

 private:
    std::vector<std::string> flag_name_map_;
    std::unordered_set<std::string> flag_names_;
};

} // namespace internal
} // namespace parthenon

parthenon::internal::UserMetadataState metadata_state;

MetadataFlag Metadata::AllocateNewFlag(std::string &&name) {
    return metadata_state.AllocateNewFlag(std::move(name));
}


std::string const &MetadataFlag::Name() const {
    return metadata_state.FlagName(*this);
}
