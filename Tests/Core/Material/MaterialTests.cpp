// This file is part of the Acts project.
//
// Copyright (C) 2017-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// clang-format off
#define BOOST_TEST_MODULE Material Tests
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
// clang-format on

#include <climits>

#include "Acts/Material/Material.hpp"
#include "Acts/Tests/CommonHelpers/FloatComparisons.hpp"
#include "Acts/Utilities/Units.hpp"

namespace Acts {
namespace Test {

using namespace Acts::UnitLiterals;

// the maximum tolerance is half the accuracy
float elMaxTolerance = 0.5 / float(UCHAR_MAX);

// first test correct boolean behavior
BOOST_AUTO_TEST_CASE(Material_boolean_test) {
  Material vacuum;
  BOOST_CHECK_EQUAL(bool(vacuum), false);

  Material something(1., 2., 3., 4., 5);
  BOOST_CHECK_EQUAL(bool(something), true);
}

// now test thge construction and units
BOOST_AUTO_TEST_CASE(Material_construction_and_units) {
  // density at room temperature
  float X0 = 9.370_cm;
  float L0 = 46.52_cm;
  float A = 28.0855;
  float Z = 14.;
  float rho = 2.329_g / std::pow(UnitConstants::cm, 3.0);

  Material silicon(X0, L0, A, Z, rho);
  CHECK_CLOSE_REL(silicon.X0(), 93.70_mm, 0.001);
  CHECK_CLOSE_REL(silicon.L0(), 465.2_mm, 0.001);
  CHECK_CLOSE_REL(silicon.Z(), 14., 0.001);
  CHECK_CLOSE_REL(silicon.A(), 28.0855, 0.001);
  CHECK_CLOSE_REL(
      silicon.rho(), 0.002329_g / std::pow(UnitConstants::cm, 3.0), 0.001);
  CHECK_CLOSE_REL(silicon.zOverAtimesRho(), 14. / 28.0855 * 0.002329, 0.0001);

  ActsVectorF<5> siliconValues;
  siliconValues << X0, L0, A, Z, rho;
  Material siliconFromValues(siliconValues);
  BOOST_CHECK_EQUAL(silicon, siliconFromValues);

  Material copiedSilicon(silicon);
  BOOST_CHECK_EQUAL(silicon, copiedSilicon);

  Material moveCopiedSilicon(std::move(copiedSilicon));
  BOOST_CHECK_EQUAL(silicon, moveCopiedSilicon);

  Material assignedSilicon = silicon;
  BOOST_CHECK_EQUAL(silicon, assignedSilicon);

  Material moveAssignedSilicon = std::move(assignedSilicon);
  BOOST_CHECK_EQUAL(silicon, moveAssignedSilicon);

  ActsVectorF<5> decomposedSilicon = silicon.classificationNumbers();
  CHECK_CLOSE_REL(decomposedSilicon, siliconValues, 1e-4);
}
}  // namespace Test
}  // namespace Acts
