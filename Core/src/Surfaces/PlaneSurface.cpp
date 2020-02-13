// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Surfaces/PlaneSurface.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "Acts/Surfaces/InfiniteBounds.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Utilities/ThrowAssert.hpp"

Acts::PlaneSurface::PlaneSurface(const PlaneSurface& other)
    : GeometryObject(), Surface(other), m_bounds(other.m_bounds) {}

Acts::PlaneSurface::PlaneSurface(const GeometryContext& gctx,
                                 const PlaneSurface& other,
                                 const Transform3D& transf)
    : GeometryObject(),
      Surface(gctx, other, transf),
      m_bounds(other.m_bounds) {}

Acts::PlaneSurface::PlaneSurface(const Vector3D& center, const Vector3D& normal)
    : Surface(), m_bounds(nullptr) {
  /// the right-handed coordinate system is defined as
  /// T = normal
  /// U = Z x T if T not parallel to Z otherwise U = X x T
  /// V = T x U
  Vector3D T = normal.normalized();
  Vector3D U = std::abs(T.dot(Vector3D::UnitZ())) < s_curvilinearProjTolerance
                   ? Vector3D::UnitZ().cross(T).normalized()
                   : Vector3D::UnitX().cross(T).normalized();
  Vector3D V = T.cross(U);
  RotationMatrix3D curvilinearRotation;
  curvilinearRotation.col(0) = U;
  curvilinearRotation.col(1) = V;
  curvilinearRotation.col(2) = T;

  // curvilinear surfaces are boundless
  Transform3D transform{curvilinearRotation};
  transform.pretranslate(center);
  Surface::m_transform = std::make_shared<const Transform3D>(transform);
}

Acts::PlaneSurface::PlaneSurface(
    const std::shared_ptr<const PlanarBounds>& pbounds,
    const Acts::DetectorElementBase& detelement)
    : Surface(detelement), m_bounds(pbounds) {
  /// surfaces representing a detector element must have bounds
  throw_assert(pbounds, "PlaneBounds must not be nullptr");
}

Acts::PlaneSurface::PlaneSurface(std::shared_ptr<const Transform3D> htrans,
                                 std::shared_ptr<const PlanarBounds> pbounds)
    : Surface(std::move(htrans)), m_bounds(std::move(pbounds)) {}

Acts::PlaneSurface& Acts::PlaneSurface::operator=(const PlaneSurface& other) {
  if (this != &other) {
    Surface::operator=(other);
    m_bounds = other.m_bounds;
  }
  return *this;
}

Acts::Surface::SurfaceType Acts::PlaneSurface::type() const {
  return Surface::Plane;
}

void Acts::PlaneSurface::localToGlobal(const GeometryContext& gctx,
                                       const Vector2D& lposition,
                                       const Vector3D& /*gmom*/,
                                       Vector3D& position) const {
  Vector3D loc3Dframe(lposition[Acts::eLOC_X], lposition[Acts::eLOC_Y], 0.);
  /// the chance that there is no transform is almost 0, let's apply it
  position = transform(gctx) * loc3Dframe;
}

bool Acts::PlaneSurface::globalToLocal(const GeometryContext& gctx,
                                       const Vector3D& position,
                                       const Vector3D& /*gmom*/,
                                       Acts::Vector2D& lposition) const {
  /// the chance that there is no transform is almost 0, let's apply it
  Vector3D loc3Dframe = (transform(gctx).inverse()) * position;
  lposition = Vector2D(loc3Dframe.x(), loc3Dframe.y());
  return ((loc3Dframe.z() * loc3Dframe.z() >
           s_onSurfaceTolerance * s_onSurfaceTolerance)
              ? false
              : true);
}

std::string Acts::PlaneSurface::name() const {
  return "Acts::PlaneSurface";
}

std::shared_ptr<Acts::PlaneSurface> Acts::PlaneSurface::clone(
    const GeometryContext& gctx, const Transform3D& shift) const {
  return std::shared_ptr<PlaneSurface>(this->clone_impl(gctx, shift));
}

Acts::PlaneSurface* Acts::PlaneSurface::clone_impl(
    const GeometryContext& gctx, const Transform3D& shift) const {
  return new PlaneSurface(gctx, *this, shift);
}

const Acts::SurfaceBounds& Acts::PlaneSurface::bounds() const {
  if (m_bounds) {
    return (*m_bounds.get());
  }
  return s_noBounds;
}

Acts::Polyhedron Acts::PlaneSurface::polyhedronRepresentation(
    const GeometryContext& gctx, size_t lseg) const {
  // Prepare vertices and faces
  std::vector<Vector3D> vertices;
  std::vector<Polyhedron::face_type> faces;
  std::vector<Polyhedron::face_type> triangularMesh;

  // If you have bounds you can create a polyhedron representation
  if (m_bounds) {
    auto vertices2D = m_bounds->vertices(lseg);
    vertices.reserve(vertices2D.size());
    for (const auto& v2D : vertices2D) {
      vertices.push_back(transform(gctx) * Vector3D(v2D.x(), v2D.y(), 0.));
    }
    // Write the face
    std::vector<size_t> face(vertices.size());
    std::iota(face.begin(), face.end(), 0);
    faces.push_back(face);
    /// Triangular mesh construction
    for (unsigned int it = 2; it < vertices.size(); ++it) {
      triangularMesh.push_back({0, it - 1, it});
    }

  } else {
    throw std::domain_error(
        "Polyhedron repr of boundless surface not possible.");
  }
  return Polyhedron(vertices, faces, triangularMesh);
}
