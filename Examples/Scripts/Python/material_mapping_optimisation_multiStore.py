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
def runMaterialMappingVariance(binMap, events, id, workDir, pipeResult):

    pathExp = os.path.join(workDir, "Mapping")
    mapName = "material-map-" + id
    mapSurface = True
    mapVolume = True

    # Create a MappingMaterialDecorator based on the tracking geometry
    matDeco = acts.IMaterialDecorator.fromFile("geometry-map.json")
    detectorTemp, trackingGeometryTemp, decoratorsTemp = getOpenDataDetector(matDeco)
    matMapDeco = acts.MappingMaterialDecorator(
        tGeometry=trackingGeometryTemp, level=acts.logging.ERROR
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
    print("Job " + str(id) + ": second pass to compute the variance", flush=True)
    # Use the material map from the previous mapping as an input
    cborMap = os.path.join(pathExp, (mapName + ".cbor"))
    matDecoVar = acts.IMaterialDecorator.fromFile(cborMap)
    detectorVar, trackingGeometryVar, decoratorsVar = getOpenDataDetector(matDecoVar)

    s = acts.examples.Sequencer(events=events, numThreads=1, logLevel=acts.logging.INFO)
    for decorator in decoratorsVar:
        s.addContextDecorator(decorator)
    wb = WhiteBoard(acts.logging.ERROR)
    context = AlgorithmContext(0, 0, wb)
    for decorator in decoratorsVar:
        assert decorator.decorate(context) == ProcessCode.SUCCESS

    # Read material step information from a ROOT TTRee
    reader = RootMaterialTrackReader(
        level=acts.logging.ERROR,
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
            config=surfaceCfg, level=acts.logging.ERROR, propagator=propagator
        )
        mmAlgCfg.materialSurfaceMapper = mapper

    if mapVolume:
        navigator = Navigator(
            trackingGeometry=trackingGeometry,
        )
        propagator = Propagator(stepper, navigator)
        mapper = VolumeMaterialMapper(
            level=acts.logging.ERROR, propagator=propagator, mappingStep=999
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

    mapping = MaterialMapping(level=acts.logging.ERROR, config=mmAlgCfg)
    s.addAlgorithm(mapping)
    s.run()

    # Compute the scoring parameters
    score = dict()

    for key in binMap:
        objective = 0
        binParameters = mapping.scoringParameters(key)
        for parameters in binParameters:
            if parameters[1] != 0:
                objective += parameters[0] / math.sqrt(
                    parameters[1]
                )  # Formula for the objective (variance/sqrt(nbTrack))
        score[key] = [dict(name="surface_score", type="objective", value=objective)]
        pipeResult.send(score)

    del mapping
    del s
    os.remove(cborMap)
    os.remove(os.path.join(pathExp, (mapName + "_tracks.root")))


# to do
def surfaceExperiment(key, nbJobs, pathDB, pipeBin, pipeResult):
    # Prepare orion experiments
    # The binning range can be changed by modifying the search space
    experiments
    storage = {
        "database": {
            "name": "database_" + str(key),
            "type": "pickleddb",
            "host": os.path.join(pathDB, "database_" + str(key) + ".pkl"),
            "timeout": 240,
        },
    }
    space = {
        "x": "uniform(1, 120, discrete=True)",
        "y": "uniform(1, 120, discrete=True)",
    }
    experiments = build_experiment(
        "s_" + str(key),
        version="1",
        space=space,
        storage=storage,
        max_idle_time=240,
    )
    trials = []
    binMap = []
    for job in range(nbJobs):
        trials.append(experiments[key].suggest())
        binMap.append(trials[key].params["x"], trials[key].params["y"])
    pipeBin.send(binMap)
    score = pipeResult.recv()
    for job in range(nbJobs):
        experiments.observe(trials[job], score[job])


if "__main__" == __name__:

    print("Starting")
    # Optimiser arguents
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--numberOfJobs", nargs="?", default=2, type=int
    )  # number of parallele jobs
    parser.add_argument(
        "--topNumberOfEvents", nargs="?", default=100, type=int
    )  # number of events per trials
    parser.add_argument(
        "--workDir", nargs="?", default=os.getcwd(), type=str
    )  # path to the work directory
    parser.add_argument(
        "--dbPath", nargs="?", default="", type=str
    )  # path to the work directory
    parser.add_argument(
        "--doPloting", nargs="?", default="False", type=bool
    )  # path to the work directory

    args = parser.parse_args()

    pathExp = os.path.join(args.workDir, "Mapping")
    pathStoreDB = os.path.join(pathExp, "Database")
    if args.dbPath == "":
        pathDB = pathStoreDB
    else:
        pathDB = args.dbPath
    pathResult = os.path.join(pathExp, "Result")

    if not os.path.isdir(pathExp):
        os.makedirs(pathExp)
    if not os.path.isdir(pathStoreDB):
        os.makedirs(pathStoreDB)
    if not os.path.isdir(pathDB):
        os.makedirs(pathDB)
    if not os.path.isdir(pathResult):
        os.makedirs(pathResult)

    # Create the tracking geometry, uses the json file to configure the proto-surfaces
    matDeco = acts.IMaterialDecorator.fromFile("geometry-map.json")
    detector, trackingGeometry, decorators = getOpenDataDetector(matDeco)

    # Use the MappingMaterialDecorator to create a binning map that can be optimised
    matMapDeco = acts.MappingMaterialDecorator(
        tGeometry=trackingGeometry, level=acts.logging.WARNING
    )
    binDict = matMapDeco.binningMap()

    from multiprocessing import Process, Pipe

    # Build one experiment per surface
    # The binning of the surfaces are independent so we split
    # an optimisation problem with a large number of variable into a lot of optimisation with 2
    binPipes_child = dict()
    resultPipes_child = dict()

    scorePipes_child = dict()
    binPipes_parent = dict()

    resultPipes_parent = dict()
    scorePipes_parent = dict()

    expJob = dict()
    OptiJob = dict()

    for key in binDict:
        binPipes_parent[key], binPipes_child[key] = Pipe()
        scorePipes_parent[key], resultPipes_child[key] = Pipe()
        expJob[key] = Process(
            target=surfaceExperiment,
            args=(
                key,
                args.numberOfJobs,
                pathDB,
                binPipes_child[key],
                scorePipes_child[key],
            ),
        )
        expJob[key].start()

    for job in range(args.numberOfJobs):
        resultPipes_parent[job], resultPipes_child[job] = Pipe()
        for key in binDict:
            binMap = dict()
            binMap[key] = binPipes_parent[key].recv()
        OptiJob[job] = Process(
            target=runMaterialMappingVariance,
            args=(
                binMap,
                args.topNumberOfEvents,
                job,
                args.workDir,
                resultPipes_child[job],
            ),
        )
        OptiJob[job].start()

    for job in range(args.numberOfJobs):
        scores = resultPipes_parent[job].recv()
        for key in binDict:
            score = scores[key]
            scorePipes_parent.send(score)
        print("Job number " + str(job) + " is over")

    # REDO THE PLOTTING PART !!!!!
    #
    # print("All the jobs are over. Now creating the optimisation plots")
    # # Create some performances plots for each surface
    # resultBinMap = dict()

    # for key in binDict:

    #     pathExpSurface = os.path.join(pathResult, "b_" + str(key))

    #     if not os.path.isdir(pathExpSurface):
    #         os.makedirs(pathExpSurface)

    #     regret = experiments[key].plot.regret()
    #     regret.write_html(pathExpSurface + "/regret.html")

    #     parallel_coordinates = experiments[key].plot.parallel_coordinates()
    #     parallel_coordinates.write_html(pathExpSurface + "/parallel_coordinates.html")

    #     lpi = experiments[key].plot.lpi()
    #     lpi.write_html(pathExpSurface + "/lpi.html")

    #     partial_dependencies = experiments[key].plot.partial_dependencies()
    #     partial_dependencies.write_html(pathExpSurface + "/partial_dependencies.html")

    #     df = experiments[key].to_pandas()
    #     best = df.iloc[df.objective.idxmin()]
    #     print(best)
    #     resultBinMap[key] = (best.x, best.y)

    # if os.path.join(pathDB, "database.pkl") != os.path.join(
    #     pathStoreDB, "database.pkl"
    # ):
    #     shutil.copyfile(
    #         os.path.join(pathDB, "database.pkl"),
    #         os.path.join(pathStoreDB, "database.pkl"),
    #     )

    # # The optimal binning has been found.
    # # Run the material mapping one last to obtain a usable material map
    # print("Running the material mapping to obtain the optimised material map")
    # matMapDeco.setBinningMap(resultBinMap)

    # # Decorate the detector with the MappingMaterialDecorator
    # resultDetector, resultTrackingGeometry, resultDecorators = getOpenDataDetector(
    #     matMapDeco
    # )

    # # Sequence for the mapping, only use one thread when mapping material
    # rMap = acts.examples.Sequencer(
    #     events=args.topNumberOfEvents, numThreads=1, logLevel=acts.logging.INFO
    # )

    # # Run the material mapping
    # from material_mapping import runMaterialMapping

    # runMaterialMapping(
    #     resultTrackingGeometry,
    #     resultDecorators,
    #     outputDir=args.workDir,
    #     inputDir=args.workDir,
    #     mapName="optimised-material-map",
    #     format=JsonFormat.Cbor,
    #     s=rMap,
    # )

    # rMap.run()
    # del rMap  # Need to be deleted to write the material map to cbor
