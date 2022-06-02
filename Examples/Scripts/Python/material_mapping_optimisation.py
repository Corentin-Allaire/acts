#!/usr/bin/env python3
import os
import argparse
import string
import math
from types import FunctionType

from orion.client import build_experiment

from acts.examples import (
    Sequencer,
    WhiteBoard,
    AlgorithmContext,
    ProcessCode,
    RootMaterialTrackReader,
    RootMaterialTrackWriter,
    MaterialMapping,
    JsonMaterialWriter,
    JsonFormat,
)

import acts
from acts import (
    Vector4,
    UnitConstants as u,
    SurfaceMaterialMapper,
    VolumeMaterialMapper,
    Navigator,
    Propagator,
    StraightLineStepper,
    MaterialMapJsonConverter,
)

from common import getOpenDataDetector


# Run the material mapping and compute the variance for each bin of each surfaces
# Return a dict with the GeometryId value of the surface as a key that stores
# a list of pairs corresponding to the variance and number of tracks associated with each bin of the surface
def runMaterialMappingVariance(binMap, events, id, workDir):

    pathExp = os.path.join(workDir, "Mapping")
    mapName = "material-map-" + id
    mapSurface = True
    mapVolume = True

    # Create a MappingMaterialDecorator based on the tracking geometry
    matDeco = acts.IMaterialDecorator.fromFile("geometry-map.json")
    detectorTemp, trackingGeometryTemp, decoratorsTemp = getOpenDataDetector(matDeco)
    matMapDeco = acts.MappingMaterialDecorator(
        tGeometry=trackingGeometryTemp, level=acts.logging.INFO
    )
    # Update the binning using the bin map corresponding to this trial
    matMapDeco.setBinningMap(binMap)

    # Decorate the detector with the MappingMaterialDecorator
    detector, trackingGeometry, decorators = getOpenDataDetector(matMapDeco)

    # Sequence for the mapping, only use one thread when mapping material
    sMap = acts.examples.Sequencer(
        events=events, numThreads=1, logLevel=acts.logging.INFO
    )

    # Run the material mapping
    from material_mapping import runMaterialMapping

    runMaterialMapping(
        trackingGeometry,
        decorators,
        outputDir=pathExp,
        inputDir=workDir,
        mapName=mapName,
        format=JsonFormat.Cbor,
        s=sMap,
    )

    sMap.run()
    del sMap  # Need to be deleted to write the material map to cbor

    # Compute the variance by rerunning the mapping

    # Use the material map from the previous mapping as an input
    cborMap = os.path.join(pathExp, (mapName + ".cbor"))
    matDecoVar = acts.IMaterialDecorator.fromFile(cborMap)
    detectorVar, trackingGeometryVar, decoratorsVar = getOpenDataDetector(matDecoVar)

    s = acts.examples.Sequencer(events=events, numThreads=1, logLevel=acts.logging.INFO)
    for decorator in decoratorsVar:
        s.addContextDecorator(decorator)
    wb = WhiteBoard(acts.logging.INFO)
    context = AlgorithmContext(0, 0, wb)
    for decorator in decoratorsVar:
        assert decorator.decorate(context) == ProcessCode.SUCCESS

    # Read material step information from a ROOT TTRee
    reader = RootMaterialTrackReader(
        level=acts.logging.INFO,
        collection="material-tracks",
        fileList=[os.path.join(workDir, "geant4_material_tracks.root")],
    )
    s.addReader(reader)

    stepper = StraightLineStepper()
    mmAlgCfg = MaterialMapping.Config(context.geoContext, context.magFieldContext)
    mmAlgCfg.trackingGeometry = trackingGeometry
    mmAlgCfg.collection = "material-tracks"

    if mapSurface:
        navigator = Navigator(
            trackingGeometry=trackingGeometry,
            resolveSensitive=True,
            resolveMaterial=True,
            resolvePassive=True,
        )
        propagator = Propagator(stepper, navigator)
        surfaceCfg = SurfaceMaterialMapper.Config(
            computeVariance=True
        )  # Don't forget to turn the `computeVariance` to true
        mapper = SurfaceMaterialMapper(
            config=surfaceCfg, level=acts.logging.INFO, propagator=propagator
        )
        mmAlgCfg.materialSurfaceMapper = mapper

    if mapVolume:
        navigator = Navigator(
            trackingGeometry=trackingGeometry,
        )
        propagator = Propagator(stepper, navigator)
        mapper = VolumeMaterialMapper(
            level=acts.logging.INFO, propagator=propagator, mappingStep=999
        )
        mmAlgCfg.materialVolumeMapper = mapper

    jmConverterCfg = MaterialMapJsonConverter.Config(
        processSensitives=True,
        processApproaches=True,
        processRepresenting=True,
        processBoundaries=True,
        processVolumes=True,
        context=context.geoContext,
    )

    mapping = MaterialMapping(level=acts.logging.INFO, config=mmAlgCfg)
    s.addAlgorithm(mapping)
    s.run()

    # Compute the scoring parameters
    score = dict()

    for key in binMap:
        score[key] = mapping.scoringParameters(key)

    del mapping
    del s
    os.remove(cborMap)
    os.remove(os.path.join(pathExp, (mapName + "_tracks.root")))
    return score


# Run `nbTrials` trials for all surfaces one after another
def runTrials(binDict, experiments, nbTrials, nbEvents, workDir):

    trials = dict()
    binMap = dict()

    for trial in range(nbTrials):
        # Get some suggested binning from the database
        # Orion use the database to prevent the same trials being run multiple times
        for key in binDict:
            trials[key] = experiments[key].suggest()
            binMap[key] = (trials[key].params["x"], trials[key].params["y"])

        # Once the binning of each surfaces has been chosen run the material mapping once with the configuration
        # Return the scoring parameters for each bin of the surface (variance, nb track)
        refID = trials[next(iter(binDict))].id  # ID of the trial for the first surface
        results = runMaterialMappingVariance(binMap, nbEvents, refID, workDir)
        # Compute a score based on the scoring parameters of each bin (variance, nb track)
        for key in binDict:
            objective = 0
            binParameters = results[key]
            for parameters in binParameters:
                if parameters[1] != 0:
                    objective += parameters[0] / math.sqrt(
                        parameters[1]
                    )  # Formula for the objective (variance/sqrt(nbTrack))
            scores = [dict(name="surface_score", type="objective", value=objective)]
            experiments[key].observe(trials[key], scores)


if "__main__" == __name__:

    # Optimiser arguents
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--numberOfJobs", nargs="?", default=2, type=int
    )  # number of parallele jobs
    parser.add_argument(
        "--numberOfTrials", nargs="?", default=1, type=int
    )  # number of trials per job
    parser.add_argument(
        "--topNumberOfEvents", nargs="?", default=100, type=int
    )  # number of events per trials
    parser.add_argument(
        "--workDir", nargs="?", default=os.getcwd(), type=str
    )  # path to the work directory

    args = parser.parse_args()

    pathExp = os.path.join(args.workDir, "Mapping")
    pathDB = os.path.join(pathExp, "Database")
    pathResult = os.path.join(pathExp, "Result")

    if not os.path.isdir(pathExp):
        os.makedirs(pathExp)
    if not os.path.isdir(pathDB):
        os.makedirs(pathDB)
    if not os.path.isdir(pathResult):
        os.makedirs(pathResult)

    # Create the tracking geometry, uses the json file to configure the proto-surfaces
    matDeco = acts.IMaterialDecorator.fromFile("geometry-map.json")
    detector, trackingGeometry, decorators = getOpenDataDetector(matDeco)

    # Use the MappingMaterialDecorator to create a binning map that can be optimised
    matMapDeco = acts.MappingMaterialDecorator(
        tGeometry=trackingGeometry, level=acts.logging.INFO
    )
    binDict = matMapDeco.binningMap()

    # Prepare orion experiments
    # The binning range can be changed by modifying the search space
    experiments = dict()
    storage = {
        "database": {
            "type": "pickleddb",
            "host": os.path.join(pathDB, "database.pkl"),
        },
    }
    space = {"x": "uniform(1, 10, discrete=True)", "y": "uniform(1, 10, discrete=True)"}

    # Build one experiment per surface
    # The binning of the surfaces are independent so we split
    # an optimisation problem with a large number of variable into a lot of optimisation with 2
    for key in binDict:
        experiments[key] = build_experiment(
            "s_" + str(key),
            version="1",
            space=space,
            storage=storage,
        )

    from multiprocessing import Process

    # Launch `numberOfJobs` optimisation jobs in parallele
    OptiJob = []
    for job in range(args.numberOfJobs):
        OptiJob.append(
            Process(
                target=runTrials,
                args=(
                    binDict,
                    experiments,
                    args.numberOfTrials,
                    args.topNumberOfEvents,
                    args.workDir,
                ),
            )
        )
        OptiJob[job].start()

    # Stop the program from going forward until all jobs are finished
    for job in range(args.numberOfJobs):
        OptiJob[job].join()

    # Create some performances plots for each surface
    for key in binDict:

        pathExpSurface = os.path.join(pathResult, "b_" + str(key))

        if not os.path.isdir(pathExpSurface):
            os.makedirs(pathExpSurface)

        regret = experiments[key].plot.regret()
        regret.write_html(pathExpSurface + "/regret.html")

        parallel_coordinates = experiments[key].plot.parallel_coordinates()
        parallel_coordinates.write_html(pathExpSurface + "/parallel_coordinates.html")

        lpi = experiments[key].plot.lpi()
        lpi.write_html(pathExpSurface + "/lpi.html")

        partial_dependencies = experiments[key].plot.partial_dependencies()
        partial_dependencies.write_html(pathExpSurface + "/partial_dependencies.html")

        df = experiments[key].to_pandas()
        best = df.iloc[df.objective.idxmin()]
        print(best)
