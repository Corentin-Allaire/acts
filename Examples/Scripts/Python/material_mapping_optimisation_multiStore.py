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


def materialMapping(
    trackingGeometry,
    decorators,
    outputDir,
    inputDir,
    mapName="material-map",
    mapSurface=True,
    mapVolume=True,
    format=JsonFormat.Json,
    s=None,
):
    s = s or Sequencer(numThreads=1)

    for decorator in decorators:
        s.addContextDecorator(decorator)

    wb = WhiteBoard(acts.logging.INFO)

    context = AlgorithmContext(0, 0, wb)

    for decorator in decorators:
        assert decorator.decorate(context) == ProcessCode.SUCCESS

    # Read material step information from a ROOT TTRee
    s.addReader(
        RootMaterialTrackReader(
            level=acts.logging.INFO,
            collection="material-tracks",
            fileList=[os.path.join(inputDir, "geant4_material_tracks.root")],
        )
    )

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
        mapper = SurfaceMaterialMapper(level=acts.logging.INFO, propagator=propagator)
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

    jmw = JsonMaterialWriter(
        level=acts.logging.VERBOSE,
        converterCfg=jmConverterCfg,
        fileName=os.path.join(outputDir, mapName),
        writeFormat=format,
    )

    mmAlgCfg.materialWriters = [jmw]

    s.addAlgorithm(MaterialMapping(level=acts.logging.INFO, config=mmAlgCfg))

    return s


# Run the material mapping and compute the variance for each bin of each surfaces
# Return a dict with the GeometryId value of the surface as a key that stores
# a list of pairs corresponding to the variance and number of tracks associated with each bin of the surface
def runMaterialMappingVariance(binMap, events, job, workDir, pipeResult):

    pathExp = os.path.join(workDir, "Mapping")
    mapName = "material-map-" + str(job)
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

    materialMapping(
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
    print("Job " + str(job) + ": second pass to compute the variance", flush=True)
    # Use the material map from the previous mapping as an input
    cborMap = os.path.join(pathExp, (mapName + ".cbor"))
    matDecoVar = acts.IMaterialDecorator.fromFile(cborMap)
    detectorVar, trackingGeometryVar, decoratorsVar = getOpenDataDetector(matDecoVar)

    sVar = acts.examples.Sequencer(
        events=events, numThreads=1, logLevel=acts.logging.INFO
    )

    materialMapping(
        trackingGeometryVar,
        decoratorsVar,
        outputDir=pathExp,
        inputDir=workDir,
        mapName=mapName,
        format=JsonFormat.Cbor,
        s=sVar,
    )

    sVar.run()

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
def surfaceExperiment(key, nbJobs, pathDB, pathResult, pipeBin, pipeResult, doPloting):
    # Prepare orion experiments
    # The binning range can be changed by modifying the search space
    storage = {
        "database": {
            "name": "database_" + str(key),
            "type": "pickleddb",
            "host": os.path.join(pathDB, "database_" + str(key) + ".pkl"),
            "timeout": 2400,
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
        max_idle_time=2400,
    )
    trials = dict()
    binMap = dict()
    for job in range(nbJobs):
        trials[job] = experiments.suggest()
        binMap[job] = (trials[job].params["x"], trials[job].params["y"])
        pipeBin.send(binMap[job])
    print("Binning for surface " + str(key) + " has been sent", flush=True)
    for job in range(nbJobs):
        score = pipeResult.recv()
        print(
            "Recieved score for job " + str(job) + " and surface " + str(key),
            flush=True,
        )
        experiments.observe(trials[job], score)
        print(
            "Score for job "
            + str(job)
            + " and surface "
            + str(key)
            + " has been written",
            flush=True,
        )

    if doPloting:
        print("All the jobs are over. Now creating the optimisation plots", flush=True)
        # Create some performances plots for each surface

        pathExpSurface = os.path.join(pathResult, "b_" + str(key))

        if not os.path.isdir(pathExpSurface):
            os.makedirs(pathExpSurface)

        regret = experiments.plot.regret()
        regret.write_html(pathExpSurface + "/regret.html")

        parallel_coordinates = experiments.plot.parallel_coordinates()
        parallel_coordinates.write_html(pathExpSurface + "/parallel_coordinates.html")

        lpi = experiments.plot.lpi()
        lpi.write_html(pathExpSurface + "/lpi.html")

        partial_dependencies = experiments.plot.partial_dependencies()
        partial_dependencies.write_html(pathExpSurface + "/partial_dependencies.html")

        df = experiments.to_pandas()
        best = df.iloc[df.objective.idxmin()]
        print(best)
        resultBinMap = (best.x, best.y)
        pipeBin.send(resultBinMap)

    # if os.path.join(pathDB, "database.pkl") != os.path.join(
    #     pathStoreDB, "database.pkl"
    # ):
    #     shutil.copyfile(
    #         os.path.join(pathDB, "database.pkl"),
    #         os.path.join(pathStoreDB, "database.pkl"),
    #     )


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
        "--doPloting", action="store_true"
    )  # path to the work directory
    parser.set_defaults(doPloting=False)

    args = parser.parse_args()

    # CHANGE PATH TO A DICT
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
        scorePipes_parent[key], scorePipes_child[key] = Pipe()
        expJob[key] = Process(
            target=surfaceExperiment,
            args=(
                key,
                args.numberOfJobs,
                pathDB,
                pathResult,
                binPipes_child[key],
                scorePipes_child[key],
                args.doPloting,
            ),
        )
        expJob[key].start()

    for job in range(args.numberOfJobs):
        resultPipes_parent[job], resultPipes_child[job] = Pipe()
        binMap = dict()
        for key in binDict:
            binMap[key] = binPipes_parent[key].recv()
        print(
            "Binning for job"
            + str(job)
            + "have been selected, now running the mapping",
            flush=True,
        )
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
        print("Job number " + str(job) + " is over", flush=True)

    if args.doPloting:
        # The optimal binning has been found.
        # Run the material mapping one last to obtain a usable material map
        print(
            "Running the material mapping to obtain the optimised material map",
            flush=True,
        )
        resultBinMap = dict()
        for key in binDict:
            resultBinMap[key] = binPipes_parent[key].recv()
        matMapDeco.setBinningMap(resultBinMap)

        # Decorate the detector with the MappingMaterialDecorator
        resultDetector, resultTrackingGeometry, resultDecorators = getOpenDataDetector(
            matMapDeco
        )

        # Sequence for the mapping, only use one thread when mapping material
        rMap = acts.examples.Sequencer(
            events=args.topNumberOfEvents, numThreads=1, logLevel=acts.logging.INFO
        )

        # Run the material mapping
        from material_mapping import runMaterialMapping

        runMaterialMapping(
            resultTrackingGeometry,
            resultDecorators,
            outputDir=args.workDir,
            inputDir=args.workDir,
            mapName="optimised-material-map",
            format=JsonFormat.Cbor,
            s=rMap,
        )

        rMap.run()
        del rMap  # Need to be deleted to write the material map to cbor
