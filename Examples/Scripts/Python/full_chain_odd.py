#!/usr/bin/env python3
import argparse
import pathlib, acts, acts.examples
import acts.examples.dd4hep
from common import getOpenDataDetector, getOpenDataDetectorDirectory
from pythia8 import addPythia8
from pathlib import Path

u = acts.UnitConstants
outputDir = pathlib.Path.cwd() / "odd_output"
outputDir.mkdir(exist_ok=True)

oddDir = getOpenDataDetectorDirectory()

oddMaterialMap = oddDir / "data/odd-material-maps.root"
oddDigiConfig = oddDir / "config/odd-digi-smearing-config.json"
oddSeedingSel = oddDir / "config/odd-seeding-config.json"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

detector, trackingGeometry, decorators = getOpenDataDetector(mdecorator=oddMaterialDeco)
field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

from particle_gun import addParticleGun, MomentumConfig, EtaConfig, ParticleConfig
from fatras import addFatras
from digitization import addDigitization
from seeding import addSeeding, SeedingAlgorithm, TruthSeedRanges
from ckf_tracks import addCKFTracks

s = acts.examples.Sequencer(events=1000, numThreads=2, logLevel=acts.logging.INFO)

evGen = addPythia8(s, rnd, hardProcess = ["Top:qqbar2ttbar=on"], npileup=200)

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

s = addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=oddDigiConfig,
    # outputDirRoot=outputDir,
    outputDirCsv=outputDir,
    rnd=rnd,
)
s = addSeeding(
    s,
    trackingGeometry,
    field,
    TruthSeedRanges(pt=(1.0*u.GeV, None), eta=(-2.7, 2.7), nHits=(9, None)),
    # seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
    geoSelectionConfigFile=oddSeedingSel,
    # outputDirRoot=outputDir,
    initialVarInflation=[100, 100, 100, 100, 100, 100],
)
s = addCKFTracks(
    s,
    trackingGeometry,
    field,
    TruthSeedRanges(pt=(1.0 * u.GeV, None), nHits=(9, None)),
    # outputDirRoot=outputDir,
    outputDirCsv=outputDir,
)

s.run()
