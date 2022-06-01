#!/usr/bin/env python3
import os
import argparse
import dataclasses
import functools
import string

import psutil
import math

from ctypes import c_uint64

from types import FunctionType

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


def runMaterialMappingVariance(binMap, events, id, pathExp):

    inputDir=os.getcwd()
    mapName="material-map-"+id 
    mapSurface=True
    mapVolume=True

    print("start")

    print(binMap)    
    matDeco = acts.IMaterialDecorator.fromFile("geometry-map.json")
    detectorTemp, trackingGeometryTemp, decoratorsTemp = getOpenDataDetector(matDeco)

    matMapDeco = acts.MappingMaterialDecorator(tGeometry=trackingGeometryTemp, level=acts.logging.INFO)
    matMapDeco.setBinningMap(binMap)

    detector, trackingGeometry, decorators = getOpenDataDetector(matMapDeco)

    sMap = acts.examples.Sequencer(
        events=events, numThreads=1, logLevel=acts.logging.INFO
    )

    from material_mapping import runMaterialMapping

    runMaterialMapping(
        trackingGeometry, decorators, outputDir=pathExp, inputDir=os.getcwd(), mapName=mapName, format=JsonFormat.Cbor, s=sMap,
    )

    sMap.run()
    del sMap
 
    # Compute the variance by rerunning the mapping

    cborMap = os.path.join(pathExp, (mapName+'.cbor'))

    matDecoVar = acts.IMaterialDecorator.fromFile(cborMap)
    detectorVar, trackingGeometryVar, decoratorsVar = getOpenDataDetector(matDecoVar)

    s = acts.examples.Sequencer(
        events=events, numThreads=1, logLevel=acts.logging.INFO
    )

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
            fileList=[os.path.join(inputDir, "geant4_material_tracks.root")],
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
        surfaceCfg = SurfaceMaterialMapper.Config(computeVariance = True)
        mapper = SurfaceMaterialMapper(config = surfaceCfg, level=acts.logging.INFO, propagator=propagator)
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
        fileName=os.path.join(pathExp, mapName),
        writeFormat=JsonFormat.Cbor,
    )

    # mmAlgCfg.materialWriters = [jmw]

    mapping = MaterialMapping(level=acts.logging.INFO, config=mmAlgCfg)
    s.addAlgorithm(mapping)

    s.run()

    # COMPUTE SCORE
    score = dict()

    for key in binMap:
        score[key] = mapping.scoringParameters(key)      

    del mapping
    del s
    os.remove(cborMap)
    os.remove(os.path.join(pathExp, (mapName+'_tracks.root'))) 
    return score

if "__main__" == __name__:

    parser = argparse.ArgumentParser()
    parser.add_argument('--numberOfTrials', nargs='?', default=25, type=int)
    parser.add_argument('--topNumberOfEvents', nargs='?', default=1000, type=int)

    args = parser.parse_args()
    pathExp = os.path.join(os.getcwd(), "Mapping")
    if not os.path.isdir(pathExp):
        os.makedirs(pathExp)

    matDeco = acts.IMaterialDecorator.fromFile("geometry-map.json")
    detector, trackingGeometry, decorators = getOpenDataDetector(matDeco)

    matMapDeco = acts.MappingMaterialDecorator(tGeometry=trackingGeometry, level=acts.logging.VERBOSE)

    experiments = dict()
    trials = dict()
    binMap = dict()

    binDict = matMapDeco.binningMap()

    storage = {
        "database": {
            "type": "pickleddb",
            "host": pathExp,
        },
    }  

    space = {
        "x": "uniform(1, 10, discrete=True)",
        "y": "uniform(1, 10, discrete=True)"
    }

    from orion.client import build_experiment
    import os

    for key in binDict:
        experiments[key] = build_experiment(
                                's_' + str(key),
                                version='1',
                                space=space,
                                storage=storage)

    for trial in range(args.numberOfTrials):

        for key in binDict:
            trials[key] = experiments[key].suggest()
            binMap[key] = (trials[key].params['x'], trials[key].params['y'])                   
        
        refID =  trials[next(iter(binDict))].id # ID of the trial for the first surface 
        scores = runMaterialMappingVariance(binMap, args.topNumberOfEvents, refID, pathExp)
        for key in binDict:
            objective = 0         
            binParameters = scores[key]
            for parameters in binParameters:
                if (parameters[1] != 0):
                    objective += parameters[0] / math.sqrt(parameters[1])      
            results = [dict(
                name='surface_score',
                type='objective',
                value=objective)]
            experiments[key].observe(trials[key], results)

    for key in binDict:

        pathExpSurface = os.path.join(pathExp, "b_"+str(key))
        if not os.path.isdir(pathExpSurface):
            os.makedirs(pathExpSurface)


        regret = experiments[key].plot.regret()
        regret.write_html( pathExpSurface + "/regret.html")
        print("1")
        parallel_coordinates = experiments[key].plot.parallel_coordinates()
        parallel_coordinates.write_html(pathExpSurface + "/parallel_coordinates.html")
        print("2")
        lpi = experiments[key].plot.lpi()
        lpi.write_html(pathExpSurface + "/lpi.html")
        print("3")
        partial_dependencies= experiments[key].plot.partial_dependencies()
        partial_dependencies.write_html(pathExpSurface + "/partial_dependencies.html")
        print("4")
        df = experiments[key].to_pandas()
        print("5")
        best = df.iloc[df.objective.idxmin()]
        print(best)

