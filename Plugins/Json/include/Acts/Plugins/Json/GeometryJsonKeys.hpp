// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>

namespace Acts {

/// @struct jsonKey
///
/// @brief store in a single place the different key used for the material
/// mapping
struct jsonKey {
  /// The name identification
  std::string namekey = "Name";
  /// The bin key
  std::string binkey = "binUtility";
  /// The local to global transformation key
  std::string transfokeys = "transformation";
  /// The type key -> proto, else
  std::string typekey = "type";
  /// The data key
  std::string datakey = "data";
  /// The geoid key
  std::string geometryidkey = "Geoid";
  /// The mapping key, add surface to mapping procedure if true
  std::string mapkey = "mapMaterial";
  /// The surface type key
  std::string surfacetypekey = "stype";
  /// The surface position key
  std::string surfacepositionkey = "sposition";
  /// The surface range key
  std::string surfacerangekey = "srange";
};

}  // namespace Acts