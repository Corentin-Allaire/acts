// This file is part of the Acts project.
//
// Copyright (C) 2019-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/TrackParametrization.hpp"
#include "ActsExamples/EventData/Trajectories.hpp"
#include "ActsExamples/Framework/WriterT.hpp"

#include <mutex>
#include <vector>

class TFile;
class TTree;

namespace ActsExamples {

/// @class RootTrajectorySummaryWriter
///
/// Write out the information (including number of measurements, outliers, holes
/// etc. and fitted track parameters) of the reconstructed trajectories into a
/// TTree
///
/// Safe to use from multiple writer threads - uses a std::mutex lock.
///
/// Each entry in the TTree corresponds to all reconstructed trajectories in one
/// single event. The event number is part of the written data.
///
/// A common file can be provided for to the writer to attach his TTree,
/// this is done by setting the Config::rootFile pointer to an existing
/// file
///
/// Safe to use from multiple writer threads - uses a std::mutex lock.
class RootTrajectorySummaryWriter final
    : public WriterT<TrajectoriesContainer> {
 public:
  struct Config {
    /// Input (fitted) trajectories collection
    std::string inputTrajectories;
    /// Input hit-particles map collection.
    std::string inputMeasurementParticlesMap;
    /// output directory.
    std::string outputDir;
    /// output filename.
    std::string outputFilename = "tracksummary.root";
    /// name of the output tree.
    std::string outputTreename = "tracksummary";
    /// file access mode.
    std::string fileMode = "RECREATE";
    /// common root file.
    TFile* rootFile = nullptr;
  };

  /// Constructor
  ///
  /// @param cfg Configuration struct
  /// @param level Message level declaration
  RootTrajectorySummaryWriter(const Config& cfg, Acts::Logging::Level lvl);
  ~RootTrajectorySummaryWriter() final override;

  /// End-of-run hook
  ProcessCode endRun() final override;

 protected:
  /// @brief Write method called by the base class
  /// @param [in] ctx is the algorithm context for event information
  /// @param [in] trajectories are what to be written out
  ProcessCode writeT(const AlgorithmContext& ctx,
                     const TrajectoriesContainer& trajectories) final override;

 private:
  Config m_cfg;             ///< The config class
  std::mutex m_writeMutex;  ///< Mutex used to protect multi-threaded writes
  TFile* m_outputFile{nullptr};  ///< The output file
  TTree* m_outputTree{nullptr};  ///< The output tree
  uint32_t m_eventNr{0};         ///< The event number
  std::vector<uint32_t>
      m_multiTrajNr;  ///< The multi-trajectory numbers in event
  std::vector<unsigned int>
      m_subTrajNr;  ///< The multi-trajectory sub-trajectory number in event

  std::vector<unsigned int> m_nStates;        ///< The number of states
  std::vector<unsigned int> m_nMeasurements;  ///< The number of measurements
  std::vector<unsigned int> m_nOutliers;      ///< The number of outliers
  std::vector<unsigned int> m_nHoles;         ///< The number of holes
  std::vector<float> m_chi2Sum;               ///< The total chi2
  std::vector<unsigned int>
      m_NDF;  ///< The number of ndf of the measurements+outliers
  std::vector<std::vector<double>>
      m_measurementChi2;  ///< The chi2 on all measurement states
  std::vector<std::vector<double>>
      m_outlierChi2;  ///< The chi2 on all outlier states
  std::vector<std::vector<double>>
      m_measurementVolume;  ///< The volume id of the measurements
  std::vector<std::vector<double>>
      m_measurementLayer;  ///< The layer id of the measurements
  std::vector<std::vector<double>>
      m_outlierVolume;  ///< The volume id of the outliers
  std::vector<std::vector<double>>
      m_outlierLayer;  ///< The layer id of the outliers

  std::vector<unsigned int>
      m_nMajorityHits;  ///< The number of hits from majority particle
  std::vector<uint64_t>
      m_majorityParticleId;  ///< The particle Id of the majority particle

  std::vector<bool> m_hasFittedParams;  ///< If the track has fitted parameter
  std::vector<float>
      m_eLOC0_fit;  ///< Fitted parameters eBoundLoc0 of all tracks in event
  std::vector<float>
      m_eLOC1_fit;  ///< Fitted parameters eBoundLoc1 of all tracks in event
  std::vector<float>
      m_ePHI_fit;  ///< Fitted parameters ePHI of all tracks in event
  std::vector<float>
      m_eTHETA_fit;  ///< Fitted parameters eTHETA of all tracks in event
  std::vector<float>
      m_eQOP_fit;  ///< Fitted parameters eQOP of all tracks in event
  std::vector<float> m_eT_fit;  ///< Fitted parameters eT of all tracks in event
  std::vector<float>
      m_err_eLOC0_fit;  ///< Fitted parameters eLOC err of all tracks in event
  std::vector<float> m_err_eLOC1_fit;  ///< Fitted parameters eBoundLoc1 err of
                                       ///< all tracks in event
  std::vector<float>
      m_err_ePHI_fit;  ///< Fitted parameters ePHI err of all tracks in event
  std::vector<float> m_err_eTHETA_fit;  ///< Fitted parameters eTHETA err of all
                                        ///< tracks in event
  std::vector<float>
      m_err_eQOP_fit;  ///< Fitted parameters eQOP err of all tracks in event
  std::vector<float>
      m_err_eT_fit;  ///< Fitted parameters eT err of all tracks in event
};

}  // namespace ActsExamples
