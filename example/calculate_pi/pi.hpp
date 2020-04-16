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
#ifndef EXAMPLE_CALCULATE_PI_PI_HPP_
#define EXAMPLE_CALCULATE_PI_PI_HPP_

// Standard Includes
#include <memory>

// Parthenon Includes
#include <parthenon/app.hpp>
#include <parthenon/driver.hpp>

namespace calculate_pi {
using namespace parthenon::app::prelude;
using namespace parthenon::driver::prelude;

/**
 * @brief Constructs a driver which estimates PI using AMR.
 */
class CalculatePi : public Driver {
 public:
  CalculatePi(ParameterInput *pin, Mesh *pm, Outputs *pout) : Driver(pin, pm, pout) {}

  /// MakeTaskList isn't a virtual routine on `Driver`, but each driver is expected to
  /// implement it.
  TaskList MakeTaskList(MeshBlock *pmb);

  /// `Execute` cylces until simulation completion.
  DriverStatus Execute() override;
};

void SetInOrOut(Container<Real> &rc);
parthenon::AmrTag CheckRefinement(Container<Real> &rc);
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

} // namespace calculate_pi

#endif // EXAMPLE_CALCULATE_PI_PI_HPP_
