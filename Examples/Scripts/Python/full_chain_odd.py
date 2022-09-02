#!/usr/bin/env python3
import pathlib, acts, acts.examples
import acts.examples.dd4hep
from common import getOpenDataDetectorDirectory
from pythia8 import addPythia8
from pathlib import Path
from common import getOpenDataDetectorDirectory
from acts.examples.odd import getOpenDataDetector

# acts.examples.dump_args_calls(locals())  # show python binding calls

u = acts.UnitConstants
outputDir = pathlib.Path.cwd() / "odd_output"
outputDir.mkdir(exist_ok=True)

oddDir = getOpenDataDetectorDirectory()

oddMaterialMap = oddDir / "data/odd-material-maps.root"
oddDigiConfig = oddDir / "config/odd-digi-smearing-config.json"
oddSeedingSel = oddDir / "config/odd-seeding-config.json"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

detector, trackingGeometry, decorators = getOpenDataDetector(
    getOpenDataDetectorDirectory(), mdecorator=oddMaterialDeco
)
field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

from acts.examples.simulation import (
    addParticleGun,
    MomentumConfig,
    EtaConfig,
    ParticleConfig,
    addFatras,
    addDigitization,
)
from acts.examples.reconstruction import (
    addSeeding,
    addCKFTracks,
    CKFPerformanceConfig,
    addVertexFitting,
    VertexFinder,
)

s = acts.examples.Sequencer(events=100, numThreads=2, logLevel=acts.logging.INFO)

evGen = addPythia8(s, rnd, hardProcess = ["Top:qqbar2ttbar=on"], npileup=0)

s.addAlgorithm(
    acts.examples.ParticleSelector(
        level=s.config.logLevel,
        inputParticles="particles_input",
        outputParticles="particles_selected",
        removeNeutral=True,
        absEtaMax=4,
        rhoMax=4.0 * u.mm,
        ptMin=500 * u.MeV,
        )
    )

# Simulation
simAlg = acts.examples.FatrasSimulation(
    level=acts.logging.INFO,
    inputParticles="particles_selected",
    outputParticlesInitial="particles_initial",
    outputParticlesFinal="particles_final",
    outputSimHits="simhits",
    randomNumbers=rnd,
    trackingGeometry=trackingGeometry,
    magneticField=field,
    generateHitsOnSensitive=True,
)

s.addAlgorithm(simAlg)

# Output
s.addWriter(
    acts.examples.CsvParticleWriter(
        level=acts.logging.INFO,
        outputDir=str(outputDir),
        inputParticles="particles_final",
        outputStem="particles_final",
    )
)

s.addWriter(
    acts.examples.CsvSimHitWriter(
        level=acts.logging.INFO,
        inputSimHits="simhits",
        outputDir=str(outputDir),
        outputStem="hits",
    )
)

s.addWriter(
    acts.examples.CsvMultiTrajectoryWriter(
        level=acts.logging.INFO,
        inputTrajectories="trajectories",
        inputMeasurementParticlesMap="measurement_particles_map",
        outputDir=str(outputDir),
    )
)

s = addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=oddDigiConfig,
    # outputDirRoot=outputDir,
    outputDirCsv=outputDir,
    rnd=rnd,
)
addSeeding(
    s,
    trackingGeometry,
    field,
    geoSelectionConfigFile=oddSeedingSel,
    outputDirRoot=outputDir,
)
addCKFTracks(
    s,
    trackingGeometry,
    field,
    CKFPerformanceConfig(ptMin=400.0 * u.MeV, nMeasurementsMin=6),
    outputDirRoot=outputDir,
)
s.addAlgorithm(
    acts.examples.TrackSelector(
        level=acts.logging.INFO,
        inputTrackParameters="fittedTrackParameters",
        outputTrackParameters="trackparameters",
        outputTrackIndices="outputTrackIndices",
        removeNeutral=True,
        absEtaMax=2.5,
        loc0Max=4.0 * u.mm,  # rho max
        ptMin=500 * u.MeV,
    )
)
addVertexFitting(
    s,
    field,
    vertexFinder=VertexFinder.Iterative,
    outputDirRoot=outputDir,
)

s.run()
