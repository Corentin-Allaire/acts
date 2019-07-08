// This file is part of the Acts project.
//
// Copyright (C) 2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <numeric>
#include <type_traits>

#include "Acts/Seeding/IBinFinder.hpp"
#include "Acts/Seeding/SeedFilter.hpp"

namespace Acts {

template <typename external_spacepoint_t>
Seedfinder<external_spacepoint_t>::Seedfinder(
    const Acts::SeedfinderConfig<external_spacepoint_t> config)
    : m_config(std::move(config)) {
  // calculation of scattering using the highland formula
  // convert pT to p once theta angle is known
  m_config.highland = 13.6 * std::sqrt(m_config.radLengthPerSeed) *
                      (1 + 0.038 * std::log(m_config.radLengthPerSeed));
  float maxScatteringAngle = m_config.highland / m_config.minPt;
  m_config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;
  // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV and
  // millimeter
  // TODO: change using ACTS units
  m_config.pTPerHelixRadius = 300. * m_config.bFieldInZ;
  m_config.minHelixDiameter2 =
      std::pow(m_config.minPt * 2 / m_config.pTPerHelixRadius, 2);
  m_config.pT2perRadius =
      std::pow(m_config.highland / m_config.pTPerHelixRadius, 2);
}

template <typename external_spacepoint_t>
template <typename spacepoint_iterator_t>
Acts::SeedfinderState<external_spacepoint_t>
Seedfinder<external_spacepoint_t>::initState(
    spacepoint_iterator_t spBegin,
    spacepoint_iterator_t spEnd,
    std::function<
        Acts::Vector2D(const external_spacepoint_t&, float, float, float)>
        covTool,
    std::shared_ptr<Acts::IBinFinder<external_spacepoint_t>> bottomBinFinder,
    std::shared_ptr<Acts::IBinFinder<external_spacepoint_t>> topBinFinder)
    const {
  static_assert(
      std::is_same<
          typename std::iterator_traits<spacepoint_iterator_t>::value_type,
          const external_spacepoint_t*>::value,
      "Iterator does not contain type this class was templated with");
  auto state = Acts::SeedfinderState<external_spacepoint_t>();
  // setup spacepoint grid config
  SpacePointGridConfig gridConf;
  gridConf.bFieldInZ = m_config.bFieldInZ;
  gridConf.minPt = m_config.minPt;
  gridConf.rMax = m_config.rMax;
  gridConf.zMax = m_config.zMax;
  gridConf.zMin = m_config.zMin;
  gridConf.deltaRMax = m_config.deltaRMax;
  gridConf.cotThetaMax = m_config.cotThetaMax;
  // create grid with bin sizes according to the configured geometry
  std::unique_ptr<SpacePointGrid<external_spacepoint_t>> grid =
      SpacePointGridCreator::createGrid<external_spacepoint_t>(gridConf);

  // get region of interest (or full detector if configured accordingly)
  float phiMin = m_config.phiMin;
  float phiMax = m_config.phiMax;
  float zMin = m_config.zMin;
  float zMax = m_config.zMax;

  // sort by radius
  // add magnitude of beamPos to rMax to avoid excluding measurements
  // create number of bins equal to number of millimeters rMax
  // (worst case minR: configured minR + 1mm)
  size_t numRBins = (m_config.rMax + m_config.beamPos.norm());
  std::vector<std::vector<
      std::unique_ptr<const InternalSpacePoint<external_spacepoint_t>>>>
      rBins(numRBins);
  for (spacepoint_iterator_t it = spBegin; it != spEnd; it++) {
    if (*it == nullptr) {
      continue;
    }
    const external_spacepoint_t& sp = **it;
    float spX = sp.x();
    float spY = sp.y();
    float spZ = sp.z();

    if (spZ > zMax || spZ < zMin) {
      continue;
    }
    float spPhi = std::atan2(spY, spX);
    if (spPhi > phiMax || spPhi < phiMin) {
      continue;
    }

    // covariance tool provided by user
    Acts::Vector2D cov =
        covTool(sp, m_config.zAlign, m_config.rAlign, m_config.sigmaError);
    Acts::Vector3D spPosition(spX, spY, spZ);
    auto isp =
        std::make_unique<const InternalSpacePoint<external_spacepoint_t>>(
            sp, spPosition, m_config.beamPos, cov);
    // calculate r-Bin index and protect against overflow (underflow not
    // possible)
    size_t rIndex = isp->radius();
    // if index out of bounds, the SP is outside the region of interest
    if (rIndex >= numRBins) {
      continue;
    }
    rBins[rIndex].push_back(std::move(isp));
  }
  // fill rbins into grid such that each grid bin is sorted in r
  // space points with delta r < rbin size can be out of order
  for (auto& rbin : rBins) {
    for (auto& isp : rbin) {
      Acts::Vector2D spLocation(isp->phi(), isp->z());
      std::vector<
          std::unique_ptr<const InternalSpacePoint<external_spacepoint_t>>>&
          bin = grid->atPosition(spLocation);
      bin.push_back(std::move(isp));
    }
  }
  state.binnedSP = std::move(grid);
  state.bottomBinFinder = bottomBinFinder;
  state.topBinFinder = topBinFinder;
  std::array<size_t, 2> numBins = state.binnedSP->numLocalBins();
  for (size_t i = 0; i < numBins[0] * numBins[1]; i++) {
    state.outputVec.push_back({});
  }
  return std::move(state);
}

template <typename external_spacepoint_t>
void
Seedfinder<external_spacepoint_t>::createSeedsForRegion(
    SeedfinderStateIterator<external_spacepoint_t> it,
    Acts::SeedfinderState<external_spacepoint_t>& state) const {
  for (size_t spIndex = 0; spIndex < it.currentBin->size(); spIndex++) {
    const InternalSpacePoint<external_spacepoint_t>* spM =
        (*(it.currentBin))[spIndex].get();

    float rM = spM->radius();
    float zM = spM->z();
    float covrM = spM->covr();
    float covzM = spM->covz();

    std::vector<size_t>& bottomBinIndices = it.bottomBinIndices;
    std::vector<size_t>& topBinIndices = it.topBinIndices;
    std::vector<const InternalSpacePoint<external_spacepoint_t>*>
        compatBottomSP;

    // bottom space point
    for (auto bottomBinIndex : bottomBinIndices) {
      auto& bottomBin = it.grid->at(bottomBinIndex);
      for (auto& spB : bottomBin) {
        float rB = spB->radius();
        float deltaR = rM - rB;
        // if r-distance is too big, try next SP in r-sorted bin
        if (deltaR > m_config.deltaRMax) {
          continue;
        }
        // if r-distance is too small, break because bins are r-sorted
        if (deltaR < m_config.deltaRMin) {
          break;
        }
        // ratio Z/R (forward angle) of space point duplet
        float cotTheta = (zM - spB->z()) / deltaR;
        if (std::fabs(cotTheta) > m_config.cotThetaMax) {
          continue;
        }
        // check if duplet origin on z axis within collision region
        float zOrigin = zM - rM * cotTheta;
        if (zOrigin < m_config.collisionRegionMin ||
            zOrigin > m_config.collisionRegionMax) {
          continue;
        }
        compatBottomSP.push_back(spB.get());
      }
    }
    // no bottom SP found -> try next spM
    if (compatBottomSP.empty()) {
      continue;
    }

    std::vector<const InternalSpacePoint<external_spacepoint_t>*> compatTopSP;

    for (auto topBinIndex : topBinIndices) {
      auto& topBin = it.grid->at(topBinIndex);
      for (auto& spT : topBin) {
        float rT = spT->radius();
        float deltaR = rT - rM;
        // this condition is the opposite of the condition for bottom SP
        if (deltaR < m_config.deltaRMin) {
          continue;
        }
        if (deltaR > m_config.deltaRMax) {
          break;
        }

        float cotTheta = (spT->z() - zM) / deltaR;
        if (std::fabs(cotTheta) > m_config.cotThetaMax) {
          continue;
        }
        float zOrigin = zM - rM * cotTheta;
        if (zOrigin < m_config.collisionRegionMin ||
            zOrigin > m_config.collisionRegionMax) {
          continue;
        }
        compatTopSP.push_back(spT.get());
      }
    }
    if (compatTopSP.empty()) {
      continue;
    }
    // contains parameters required to calculate circle with linear equation
    // ...for bottom-middle
    std::vector<LinCircle> linCircleBottom;
    // ...for middle-top
    std::vector<LinCircle> linCircleTop;
    transformCoordinates(compatBottomSP, *spM, true, linCircleBottom);
    transformCoordinates(compatTopSP, *spM, false, linCircleTop);

    // create vectors here to avoid reallocation in each loop
    std::vector<const InternalSpacePoint<external_spacepoint_t>*> topSpVec;
    std::vector<float> curvatures;
    std::vector<float> impactParameters;

    std::vector<std::pair<
        float,
        std::unique_ptr<const InternalSeed<external_spacepoint_t>>>>
        seedsPerSpM;
    size_t numBotSP = compatBottomSP.size();
    size_t numTopSP = compatTopSP.size();

    for (size_t b = 0; b < numBotSP; b++) {
      auto lb = linCircleBottom[b];
      float Zob = lb.Zo;
      float cotThetaB = lb.cotTheta;
      float Vb = lb.V;
      float Ub = lb.U;
      float ErB = lb.Er;
      float iDeltaRB = lb.iDeltaR;

      // 1+(cot^2(theta)) = 1/sin^2(theta)
      float iSinTheta2 = (1. + cotThetaB * cotThetaB);
      // calculate max scattering for min momentum at the seed's theta angle
      // scaling scatteringAngle^2 by sin^2(theta) to convert pT^2 to p^2
      // accurate would be taking 1/atan(thetaBottom)-1/atan(thetaTop) <
      // scattering
      // but to avoid trig functions we approximate cot by scaling by
      // 1/sin^4(theta)
      // resolving with pT to p scaling --> only divide by sin^2(theta)
      // max approximation error for allowed scattering angles of 0.04 rad at
      // eta=infinity: ~8.5%
      float scatteringInRegion2 = m_config.maxScatteringAngle2 * iSinTheta2;
      // multiply the squared sigma onto the squared scattering
      scatteringInRegion2 *=
          m_config.sigmaScattering * m_config.sigmaScattering;

      // clear all vectors used in each inner for loop
      topSpVec.clear();
      curvatures.clear();
      impactParameters.clear();
      for (size_t t = 0; t < numTopSP; t++) {
        auto lt = linCircleTop[t];

        // add errors of spB-spM and spM-spT pairs and add the correlation term
        // for errors on spM
        float error2 = lt.Er + ErB +
                       2 * (cotThetaB * lt.cotTheta * covrM + covzM) *
                           iDeltaRB * lt.iDeltaR;

        float deltaCotTheta = cotThetaB - lt.cotTheta;
        float deltaCotTheta2 = deltaCotTheta * deltaCotTheta;
        float error;
        float dCotThetaMinusError2;
        // if the error is larger than the difference in theta, no need to
        // compare with scattering
        if (deltaCotTheta2 - error2 > 0) {
          deltaCotTheta = std::abs(deltaCotTheta);
          // if deltaTheta larger than the scattering for the lower pT cut, skip
          error = std::sqrt(error2);
          dCotThetaMinusError2 =
              deltaCotTheta2 + error2 - 2 * deltaCotTheta * error;
          // avoid taking root of scatteringInRegion
          // if left side of ">" is positive, both sides of unequality can be
          // squared
          // (scattering is always positive)

          if (dCotThetaMinusError2 > scatteringInRegion2) {
            continue;
          }
        }

        // protects against division by 0
        float dU = lt.U - Ub;
        if (dU == 0.) {
          continue;
        }
        // A and B are evaluated as a function of the circumference parameters
        // x_0 and y_0
        float A = (lt.V - Vb) / dU;
        float S2 = 1. + A * A;
        float B = Vb - A * Ub;
        float B2 = B * B;
        // sqrt(S2)/B = 2 * helixradius
        // calculated radius must not be smaller than minimum radius
        if (S2 < B2 * m_config.minHelixDiameter2) {
          continue;
        }
        // 1/helixradius: (B/sqrt(S2))/2 (we leave everything squared)
        float iHelixDiameter2 = B2 / S2;
        // calculate scattering for p(T) calculated from seed curvature
        float pT2scatter = 4 * iHelixDiameter2 * m_config.pT2perRadius;
        // TODO: include upper pT limit for scatter calc
        // convert p(T) to p scaling by sin^2(theta) AND scale by 1/sin^4(theta)
        // from rad to deltaCotTheta
        float p2scatter = pT2scatter * iSinTheta2;
        // if deltaTheta larger than allowed scattering for calculated pT, skip
        if (deltaCotTheta2 - error2 > 0 &&
            dCotThetaMinusError2 > p2scatter * m_config.sigmaScattering *
                                       m_config.sigmaScattering) {
          continue;
        }
        // A and B allow calculation of impact params in U/V plane with linear
        // function
        // (in contrast to having to solve a quadratic function in x/y plane)
        float Im = std::abs((A - B * rM) * rM);

        if (Im <= m_config.impactMax) {
          topSpVec.push_back(compatTopSP[t]);
          // inverse diameter is signed depending if the curvature is
          // positive/negative in phi
          curvatures.push_back(B / std::sqrt(S2));
          impactParameters.push_back(Im);
        }
      }
      if (!topSpVec.empty()) {
        std::vector<std::pair<
            float,
            std::unique_ptr<const InternalSeed<external_spacepoint_t>>>>
            sameTrackSeeds;
        sameTrackSeeds = std::move(m_config.seedFilter->filterSeeds_2SpFixed(
            *compatBottomSP[b],
            *spM,
            topSpVec,
            curvatures,
            impactParameters,
            Zob));
        seedsPerSpM.insert(
            seedsPerSpM.end(),
            std::make_move_iterator(sameTrackSeeds.begin()),
            std::make_move_iterator(sameTrackSeeds.end()));
      }
    }
    m_config.seedFilter->filterSeeds_1SpFixed(
        seedsPerSpM, state.outputVec[it.outputIndex]);
  }
}

template <typename external_spacepoint_t>
void
Seedfinder<external_spacepoint_t>::transformCoordinates(
    std::vector<const InternalSpacePoint<external_spacepoint_t>*>& vec,
    const InternalSpacePoint<external_spacepoint_t>& spM,
    bool bottom,
    std::vector<LinCircle>& linCircleVec) const {
  float xM = spM.x();
  float yM = spM.y();
  float zM = spM.z();
  float rM = spM.radius();
  float covzM = spM.covz();
  float covrM = spM.covr();
  float cosPhiM = xM / rM;
  float sinPhiM = yM / rM;
  for (auto sp : vec) {
    float deltaX = sp->x() - xM;
    float deltaY = sp->y() - yM;
    float deltaZ = sp->z() - zM;
    // calculate projection fraction of spM->sp vector pointing in same
    // direction as
    // vector origin->spM (x) and projection fraction of spM->sp vector pointing
    // orthogonal to origin->spM (y)
    float x = deltaX * cosPhiM + deltaY * sinPhiM;
    float y = deltaY * cosPhiM - deltaX * sinPhiM;
    // 1/(length of M -> SP)
    float iDeltaR2 = 1. / (deltaX * deltaX + deltaY * deltaY);
    float iDeltaR = std::sqrt(iDeltaR2);
    //
    int bottomFactor = 1 * (int(!bottom)) - 1 * (int(bottom));
    // cot_theta = (deltaZ/deltaR)
    float cot_theta = deltaZ * iDeltaR * bottomFactor;
    // VERY frequent (SP^3) access
    LinCircle l;
    l.cotTheta = cot_theta;
    // location on z-axis of this SP-duplet
    l.Zo = zM - rM * cot_theta;
    l.iDeltaR = iDeltaR;
    // transformation of circle equation (x,y) into linear equation (u,v)
    // x^2 + y^2 - 2x_0*x - 2y_0*y = 0
    // is transformed into
    // 1 - 2x_0*u - 2y_0*v = 0
    // using the following m_U and m_V
    // (u = A + B*v); A and B are created later on
    l.U = x * iDeltaR2;
    l.V = y * iDeltaR2;
    // error term for sp-pair without correlation of middle space point
    l.Er = ((covzM + sp->covz()) +
            (cot_theta * cot_theta) * (covrM + sp->covr())) *
           iDeltaR2;
    linCircleVec.push_back(l);
  }
}
}  // namespace Acts
