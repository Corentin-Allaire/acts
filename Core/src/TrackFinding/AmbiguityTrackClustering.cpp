// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/TrackFinding/detail/AmbiguityTrackClustering.hpp"

std::unordered_map<int, std::vector<int>> Acts::detail::clusterDuplicateTracks(
    const std::multimap<int, std::pair<int, std::vector<int>>>& trackMap,
    int minNbHits = 1) {
  // Unordered map associating a vector with all the track ID of a cluster to
  // the ID of the first track of the cluster
  std::unordered_map<int, std::vector<int>> cluster;
  // Unordered map associating hits to the ID of the first track of the
  // different clusters.
  std::map<int, int> hitToTrack;

  // Loop over all the tracks
  for (const auto& track : trackMap) {
    std::vector<int> hits = track.second.second;
    // Unordered map associating number of matched hits to the ID of the
    // cluster
    std::map<int, int> nbMatchedToCluster;
    auto matchedTrack = hitToTrack.end();
    auto mCluster = nbMatchedToCluster.end();
    // Loop over all the hits in the track
    for (const auto& hit : hits) {
      // Check if the hit is already associated to a track
      matchedTrack = hitToTrack.find(hit);
      if (matchedTrack != hitToTrack.end()) {
        mCluster = nbMatchedToCluster.find(matchedTrack->second);
        if (mCluster == nbMatchedToCluster.end()) {
          nbMatchedToCluster.emplace(matchedTrack->second, 1);
        } else {
          // Increase the number of matched hits to the cluster
          mCluster->second++;
        }
      }
    }
    int maxMatched = 0;
    int clusterID = -1;
    // Loop over all the matched clusters to find the one with the highest
    // number of matched hits
    for (const auto& matechedCluster : nbMatchedToCluster) {
      if (matechedCluster.second > minNbHits &&
          matechedCluster.second > maxMatched) {
        maxMatched = matechedCluster.second;
        clusterID = matechedCluster.first;
      }
    }
    // A cluster has been found
    if (clusterID != -1) {
      // Add the track to the cluster
      cluster.at(clusterID).push_back(track.second.first);
    }
    // None of the hits have been matched to a track create a new cluster
    else {
      cluster.emplace(track.second.first,
                      std::vector<int>(1, track.second.first));
      for (const auto& hit : hits) {
        // Add the hits of the new cluster to the hitToTrack
        hitToTrack.emplace(hit, track.second.first);
      }
    }
  }
  return cluster;
}
