//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
//! \file openpmd.cpp
//  \brief openPMD I/O for snapshots

#include <string>

// Parthenon headers
#include "defs.hpp"
#include "interface/variable_state.hpp"
#include "mesh/mesh.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs.hpp"

// openPMD headers
#include <openPMD/openPMD.hpp>

namespace parthenon {
using namespace OutputUtils;

void OpenPMDOutput::SetupMeshComponent(openPMD::Mesh &mesh, int meshLevel,
                                       std::string comp_name) const {
  auto global_size = getReversedVec(global_box.size());
  std::vector<double> const grid_spacing = getReversedVec(full_geom.CellSize());
  std::vector<double> const global_offset = getReversedVec(full_geom.ProbLo());

  // Prepare the type of dataset that will be written
  mesh.setDataOrder(openPMD::Mesh::DataOrder::C);
  mesh.setGridSpacing(grid_spacing);
  mesh.setGridGlobalOffset(global_offset);
  mesh.setAttribute("fieldSmoothing", "none");

  auto mesh_comp = mesh[comp_name];
  auto const dataset = openPMD::Dataset(openPMD::determineDatatype<Real>(), global_size);
  mesh_comp.resetDataset(dataset);
  std::vector<Real> relativePosition{0.5, 0.5, 0.5}; // cell-centered only (for now)
  mesh_comp.setPosition(relativePosition);
}

void OpenPMDOutput::GetMeshComponentName(int meshLevel, std::string &field_name) const {
  if (0 == meshLevel) return;
  field_name += std::string("_lvl").append(std::to_string(meshLevel));
}

//----------------------------------------------------------------------------------------
//! \fn void OpenPMDOutput:::WriteOutputFile(Mesh *pm)
//  \brief  Write all Cell variables using openPMD
void OpenPMDOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                    const SignalHandler::OutputSignal signal) {
  // open file
  auto series = openPMD::Series(output_params.file_basename, openPMD::Access::CREATE,
                                MPI_COMM_WORLD);
  series.setAuthor("Parthenon AMR framework");

  auto series_iteration = series.iterations[output_params.file_number];
  series_iteration.open();
  series_iteration.setTime(tm->time);

  // create dataset for each var
  auto mbd = pm->block_list[0]->meshblock_data.Get();
  for (const auto &var : mbd->GetVariableVector()) {
    if (!var->IsSet(Metadata::Cell)) {
      continue;
    }
    const auto var_info = VarInfo(var);

    // TODO: loop over refinement levels
    {
      auto field = series_iteration.meshes[var_info.label];
      for (int icomp = 0; icomp < var_info.num_components; ++icomp) {
        const std::string varname = var_info.component_labels.at(icomp);
        auto dataset = openPMD::Dataset(openPMD::determineDatatype<Real>(),
                                        {level_nz, level_ny, level_nx});
        field[varname].resetDataset(dataset);
      }
    }
  }

  // write chunks
  for (auto &pmb : pm->block_list) {
    auto &bounds = pmb->cellbounds;
    auto ib = bounds.GetBoundsI(IndexDomain::entire);
    auto jb = bounds.GetBoundsJ(IndexDomain::entire);
    auto kb = bounds.GetBoundsK(IndexDomain::entire);
    auto ni = ib.e - ib.s + 1;
    auto nj = jb.e - jb.s + 1;
    auto nk = kb.e - kb.s + 1;
    uint64_t ncells = ni * nj * nk;

    // TODO: how does openPMD handle ghost cells?
    auto ib_int = bounds.GetBoundsI(IndexDomain::interior);
    auto jb_int = bounds.GetBoundsJ(IndexDomain::interior);
    auto kb_int = bounds.GetBoundsK(IndexDomain::interior);

    auto &coords = pmb->coords;
    Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
    Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
    Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);
    std::array<Real, 3> corner = coords.GetXmin();

    auto &mbd = pmb->meshblock_data.Get();

    for (const auto &var : mbd->GetVariableVector()) {
      if (!var->IsSet(Metadata::Cell)) {
        continue;
      }
      const auto var_info = VarInfo(var);

      // TODO: determine which refinement level we are on
      auto field = series_iteration.meshes[var_info.label];

      // loop over field components
      for (int icomp = 0; icomp < var_info.num_components; ++icomp) {
        const std::string varname = var_info.component_labels.at(icomp);
        auto const data = Kokkos::subview(var->data, 0, 0, icomp, Kokkos::ALL(),
                                          Kokkos::ALL(), Kokkos::ALL());
        // TODO: compute chunk coordinates in global index space
        field[varname].storeChunk(data.data(), {ks, js, is}, {ke, je, ie});
      }
    }
  }

  // flush to disk
  series.flush();

  // close file
  series.close();

  // advance output parameters
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
}

} // namespace parthenon
