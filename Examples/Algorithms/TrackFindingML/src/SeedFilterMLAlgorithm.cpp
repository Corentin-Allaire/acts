// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFindingML/SeedFilterMLAlgorithm.hpp"

<<<<<<< HEAD
#include "Acts/Plugins/Mlpack/SeedFilterDBScanClustering.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
=======
#include "ActsExamples/Framework/ProcessCode.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/TrackFindingML/SeedFilterDBScanClustering.hpp"
>>>>>>> upstream/main

#include <iterator>
#include <map>

ActsExamples::SeedFilterMLAlgorithm::SeedFilterMLAlgorithm(
    ActsExamples::SeedFilterMLAlgorithm::Config cfg, Acts::Logging::Level lvl)
    : ActsExamples::IAlgorithm("SeedFilterMLAlgorithm", lvl),
      m_cfg(std::move(cfg)),
      m_seedClassifier(m_cfg.inputSeedFilterNN.c_str()) {
  if (m_cfg.inputTrackParameters.empty()) {
<<<<<<< HEAD
    throw std::invalid_argument("Missing trajectories input collection");
  }
  if (m_cfg.inputSimSeeds.empty()) {
    throw std::invalid_argument("Missing trajectories input collection");
  }
  if (m_cfg.outputTrackParameters.empty()) {
    throw std::invalid_argument("Missing trajectories output collection");
  }
  if (m_cfg.outputSimSeeds.empty()) {
    throw std::invalid_argument("Missing trajectories output collection");
=======
    throw std::invalid_argument("Missing track parameters input collection");
  }
  if (m_cfg.inputSimSeeds.empty()) {
    throw std::invalid_argument("Missing seed input collection");
  }
  if (m_cfg.outputTrackParameters.empty()) {
    throw std::invalid_argument("Missing track parameters output collection");
  }
  if (m_cfg.outputSimSeeds.empty()) {
    throw std::invalid_argument("Missing seed output collection");
>>>>>>> upstream/main
  }
  m_inputTrackParameters.initialize(m_cfg.inputTrackParameters);
  m_inputSimSeeds.initialize(m_cfg.inputSimSeeds);
  m_outputTrackParameters.initialize(m_cfg.outputTrackParameters);
  m_outputSimSeeds.initialize(m_cfg.outputSimSeeds);
}

ActsExamples::ProcessCode ActsExamples::SeedFilterMLAlgorithm::execute(
    const AlgorithmContext& ctx) const {
  // Read input data
  const auto& seeds = m_inputSimSeeds(ctx);
  const auto& params = m_inputTrackParameters(ctx);
  if (seeds.size() != params.size()) {
    throw std::invalid_argument(
<<<<<<< HEAD
        "The number of seeds and parameters is different");
=======
        "The number of seeds and track parameters is different");
>>>>>>> upstream/main
  }

  Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      networkInput(seeds.size(), 14);
<<<<<<< HEAD
  std::vector<std::vector<double>> clusteringParams;
  std::vector<int> mapSeepIndex;
  // Loop over the seed and parameters to fill the input for the clustering
  // and the NN
  for (size_t i = 0; i < seeds.size(); i++) {
    double pT = std::abs(1.0 / params[i].parameters()[Acts::eBoundQOverP]) *
                std::sin(params[i].parameters()[Acts::eBoundTheta]);
    mapSeepIndex.push_back(i);
    size_t NNindex = mapSeepIndex.size() - 1;
    double eta =
        std::atanh(std::cos(params[i].parameters()[Acts::eBoundTheta]));
    double phi = params[i].parameters()[Acts::eBoundPhi];
    // Fill the clustering input
    clusteringParams.push_back({phi, eta, seeds[i].z() / 50, pT});
    // Fill the NN input
    networkInput.row(NNindex) << pT, eta, phi, seeds[i].sp()[0]->x(),
=======
  std::vector<std::array<double, 4>> clusteringParams;
  // Loop over the seed and parameters to fill the input for the clustering
  // and the NN
  for (std::size_t i = 0; i < seeds.size(); i++) {
    // Compute the track parameters
    double pT = std::abs(1.0 / params[i].parameters()[Acts::eBoundQOverP]) *
                std::sin(params[i].parameters()[Acts::eBoundTheta]);
    double eta =
        std::atanh(std::cos(params[i].parameters()[Acts::eBoundTheta]));
    double phi = params[i].parameters()[Acts::eBoundPhi];

    // Fill and weight the clustering inputs
    clusteringParams.push_back(
        {phi / m_cfg.clusteringWeighPhi, eta / m_cfg.clusteringWeighEta,
         seeds[i].z() / m_cfg.clusteringWeighZ, pT / m_cfg.clusteringWeighPt});

    // Fill the NN input
    networkInput.row(i) << pT, eta, phi, seeds[i].sp()[0]->x(),
>>>>>>> upstream/main
        seeds[i].sp()[0]->y(), seeds[i].sp()[0]->z(), seeds[i].sp()[1]->x(),
        seeds[i].sp()[1]->y(), seeds[i].sp()[1]->z(), seeds[i].sp()[2]->x(),
        seeds[i].sp()[2]->y(), seeds[i].sp()[2]->z(), seeds[i].z(),
        seeds[i].seedQuality();
  }

  // Cluster the tracks using DBscan
  auto cluster = Acts::dbscanSeedClustering(
      clusteringParams, m_cfg.epsilonDBScan, m_cfg.minPointsDBScan);

  // Select the ID of the track we want to keep
<<<<<<< HEAD
  std::vector<int> goodSeed =
      m_seedClassifier.solveAmbiguity(cluster, networkInput);
=======
  std::vector<std::size_t> goodSeed = m_seedClassifier.solveAmbiguity(
      cluster, networkInput, m_cfg.minSeedScore);
>>>>>>> upstream/main

  // Create the output seed collection
  SimSeedContainer outputSeeds;
  outputSeeds.reserve(goodSeed.size());

  // Create the output track parameters collection
  TrackParametersContainer outputTrackParameters;
  outputTrackParameters.reserve(goodSeed.size());

<<<<<<< HEAD
  for (auto&& i : goodSeed) {
    outputSeeds.push_back(seeds[mapSeepIndex[i]]);
    outputTrackParameters.push_back(params[mapSeepIndex[i]]);
=======
  for (auto i : goodSeed) {
    outputSeeds.push_back(seeds[i]);
    outputTrackParameters.push_back(params[i]);
>>>>>>> upstream/main
  }

  m_outputSimSeeds(ctx, SimSeedContainer{outputSeeds});
  m_outputTrackParameters(ctx, TrackParametersContainer{outputTrackParameters});

  return ActsExamples::ProcessCode::SUCCESS;
}
