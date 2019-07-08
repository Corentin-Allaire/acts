// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// LineBounds.cpp, Acts project
///////////////////////////////////////////////////////////////////

#include "Acts/Surfaces/LineBounds.hpp"

#include <iomanip>
#include <iostream>

Acts::LineBounds::LineBounds(double radius, double halez)
    : m_radius(std::abs(radius)), m_halfZ(std::abs(halez)) {}

Acts::LineBounds::~LineBounds() = default;

Acts::LineBounds*
Acts::LineBounds::clone() const {
  return new LineBounds(*this);
}

Acts::SurfaceBounds::BoundsType
Acts::LineBounds::type() const {
  return SurfaceBounds::Line;
}

std::vector<TDD_real_t>
Acts::LineBounds::valueStore() const {
  std::vector<TDD_real_t> values(LineBounds::bv_length);
  values[LineBounds::bv_radius] = r();
  values[LineBounds::bv_halfZ] = halflengthZ();
  return values;
}

bool
Acts::LineBounds::inside(
    const Acts::Vector2D& lpos,
    const Acts::BoundaryCheck& bcheck) const {
  return bcheck.isInside(
      lpos, Vector2D(0, -halflengthZ()), Vector2D(r(), halflengthZ()));
}

double
Acts::LineBounds::distanceToBoundary(const Acts::Vector2D& lpos) const {
  // per definition the min Distance of a correct local position is r
  return lpos[Acts::eLOC_R];
}

// ostream operator overload
std::ostream&
Acts::LineBounds::toStream(std::ostream& sl) const {
  sl << std::setiosflags(std::ios::fixed);
  sl << std::setprecision(7);
  sl << "Acts::LineBounds: (radius, halflengthInZ) = ";
  sl << "(" << r() << ", " << halflengthZ() << ")";
  sl << std::setprecision(-1);
  return sl;
}
