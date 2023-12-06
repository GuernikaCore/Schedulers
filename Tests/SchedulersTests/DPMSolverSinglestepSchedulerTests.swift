//
//  DPMSolverSinglestepSchedulerTests.swift
//
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import XCTest
import CoreML
import RandomGenerator
@testable import Schedulers

final class DPMSolverSinglestepSchedulerTests: XCTestCase {
    func test1Steps() throws {
        var scheduler = DPMSolverSinglestepScheduler(stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [999])
        scheduler = DPMSolverSinglestepScheduler(strength: 0.9, stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [999])
    }
    
    func test2Steps() throws {
        let scheduler = DPMSolverSinglestepScheduler(stepCount: 2)
        XCTAssertEqual(scheduler.timeSteps, [999,   500])
    }
    
    func test33Steps() throws {
        let scheduler = DPMSolverSinglestepScheduler(stepCount: 33)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 969, 938, 908, 878, 848, 817, 787, 757, 727, 696, 666, 636, 605, 575, 545, 515, 484, 454, 424, 394, 363,
            333, 303, 272, 242, 212, 182, 151, 121,  91,  61,  30
        ])
    }
    
    func test33StepsKarras() throws {
        let scheduler = DPMSolverSinglestepScheduler(stepCount: 33, useKarrasSigmas: true)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 977, 954, 930, 905, 878, 849, 819, 786, 751, 714, 673, 630, 584, 535, 482, 427, 371, 315, 259, 207, 160,
            119,  86,  59,  39,  25,  15,   9,   5,   2,   1,   0
        ])
    }
    
    func test50Steps() throws {
        var scheduler = DPMSolverSinglestepScheduler(stepCount: 50)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659, 639, 619, 599, 579,
            559, 539, 519, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160, 140,
            120, 100,  80,  60,  40,  20
        ])
        scheduler = DPMSolverSinglestepScheduler(strength: 0.7, stepCount: 50)
        XCTAssertEqual(scheduler.timeSteps, [
            699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300, 280,
            260, 240, 220, 200, 180, 160, 140, 120, 100,  80,  60,  40,  20
        ])
    }
    
    func test50StepsKarras() throws {
        var scheduler = DPMSolverSinglestepScheduler(stepCount: 50, useKarrasSigmas: true)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 985, 970, 955, 940, 924, 907, 890, 871, 853, 833, 813, 791, 769, 746, 721, 696, 669, 641, 612, 581, 549,
            516, 481, 446, 409, 372, 335, 299, 263, 228, 195, 165, 137, 112,  90,  71,  55,  42,  32,  23,  17,  12,   8,
            6,   4,   2,   1,   1,   0
        ])
        scheduler = DPMSolverSinglestepScheduler(strength: 0.7, stepCount: 50, useKarrasSigmas: true)
        XCTAssertEqual(scheduler.timeSteps, [
            721, 696, 669, 641, 612, 581, 549, 516, 481, 446, 409, 372, 335, 299, 263, 228, 195, 165, 137, 112,  90,  71,
            55,  42,  32,  23,  17,  12,   8,   6,   4,   2,   1,   1,   0
        ])
    }
    
    func testStepEpsilon() throws {
        let scheduler = DPMSolverSinglestepScheduler(stepCount: 20)
        let generator: RandomGenerator = TorchRandomGenerator(seed: 50)
        let stdev = scheduler.initNoiseSigma
        var latent: MLShapedArray<Float32> = generator.nextArray(shape: [1, 4, 64, 64], mean: 0, stdev: stdev)
        zip(latent[0][0][0][0..<10].scalars, [
            -1.1588,  0.3673,  0.7110, -0.2373, -1.0129,  0.5580, -0.8784, -1.1446, -0.7629, -0.0860
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        let expectedOutputs: [[Float32]] = [
            [-1.1576,  0.3669,  0.7102, -0.2371, -1.0118,  0.5574, -0.8774, -1.1434, -0.7621, -0.0859],
            [-1.1565,  0.3666,  0.7096, -0.2369, -1.0109,  0.5569, -0.8767, -1.1424, -0.7614, -0.0859],
            [-1.1564,  0.3666,  0.7095, -0.2368, -1.0108,  0.5568, -0.8765, -1.1422, -0.7613, -0.0859],
            [-1.1530,  0.3655,  0.7074, -0.2362, -1.0079,  0.5552, -0.8740, -1.1389, -0.7591, -0.0856],
            [-1.1537,  0.3657,  0.7079, -0.2363, -1.0085,  0.5555, -0.8745, -1.1396, -0.7596, -0.0857],
            [-1.1468,  0.3635,  0.7036, -0.2349, -1.0024,  0.5522, -0.8693, -1.1328, -0.7550, -0.0851],
            [-1.1493,  0.3643,  0.7052, -0.2354, -1.0047,  0.5534, -0.8712, -1.1353, -0.7567, -0.0853],
            [-1.1371,  0.3605,  0.6977, -0.2329, -0.9940,  0.5475, -0.8619, -1.1232, -0.7486, -0.0844],
            [-1.1431,  0.3624,  0.7014, -0.2341, -0.9992,  0.5504, -0.8665, -1.1291, -0.7526, -0.0849],
            [-1.1244,  0.3564,  0.6899, -0.2303, -0.9829,  0.5414, -0.8523, -1.1107, -0.7403, -0.0835],
            [-1.1352,  0.3599,  0.6965, -0.2325, -0.9923,  0.5466, -0.8605, -1.1213, -0.7474, -0.0843],
            [-1.1083,  0.3513,  0.6800, -0.2270, -0.9687,  0.5336, -0.8401, -1.0947, -0.7296, -0.0823],
            [-1.1259,  0.3569,  0.6908, -0.2306, -0.9842,  0.5422, -0.8535, -1.1122, -0.7413, -0.0836],
            [-1.0907,  0.3457,  0.6692, -0.2234, -0.9534,  0.5252, -0.8267, -1.0773, -0.7181, -0.0810],
            [-1.1152,  0.3535,  0.6842, -0.2284, -0.9748,  0.5370, -0.8453, -1.1015, -0.7342, -0.0828],
            [-1.0722,  0.3399,  0.6579, -0.2196, -0.9372,  0.5163, -0.8127, -1.0591, -0.7059, -0.0796],
            [-1.1017,  0.3492,  0.6759, -0.2256, -0.9630,  0.5305, -0.8351, -1.0882, -0.7253, -0.0818],
            [-1.0522,  0.3335,  0.6456, -0.2155, -0.9197,  0.5067, -0.7976, -1.0393, -0.6927, -0.0781],
            [-1.0739,  0.3404,  0.6589, -0.2199, -0.9387,  0.5171, -0.8140, -1.0607, -0.7070, -0.0797],
            [-1.1136,  0.3530,  0.6833, -0.2281, -0.9734,  0.5362, -0.8441, -1.1000, -0.7332, -0.0827],
        ]
        
        var output = latent
        for (index, t) in scheduler.timeSteps.enumerated() {
            latent = scheduler.scaleModelInput(timeStep: t, sample: latent)
            output = scheduler.step(output: output, timeStep: t, sample: latent, generator: generator)
            
            zip(output[0][0][0][0..<10].scalars, expectedOutputs[index]).forEach { actual, expected in
                XCTAssertEqual(actual, expected, accuracy: 0.02)
            }
        }
    }
    
    func testStepEpsilonKarras() throws {
        let scheduler = DPMSolverSinglestepScheduler(stepCount: 20, useKarrasSigmas: true)
        let generator: RandomGenerator = TorchRandomGenerator(seed: 50)
        let stdev = scheduler.initNoiseSigma
        var latent: MLShapedArray<Float32> = generator.nextArray(shape: [1, 4, 64, 64], mean: 0, stdev: stdev)
        zip(latent[0][0][0][0..<10].scalars, [
            -1.1588,  0.3673,  0.7110, -0.2373, -1.0129,  0.5580, -0.8784, -1.1446, -0.7629, -0.0860
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        let expectedOutputs: [[Float32]] = [
            [-1.1580,  0.3671,  0.7105, -0.2372, -1.0122,  0.5576, -0.8777, -1.1438, -0.7624, -0.0860],
            [-1.1572,  0.3668,  0.7100, -0.2370, -1.0115,  0.5572, -0.8771, -1.1430, -0.7618, -0.0859],
            [-1.1570,  0.3668,  0.7099, -0.2370, -1.0114,  0.5571, -0.8770, -1.1429, -0.7617, -0.0859],
            [-1.1543,  0.3659,  0.7082, -0.2364, -1.0090,  0.5558, -0.8750, -1.1402, -0.7600, -0.0857],
            [-1.1540,  0.3658,  0.7080, -0.2363, -1.0087,  0.5557, -0.8747, -1.1399, -0.7597, -0.0857],
            [-1.1464,  0.3634,  0.7034, -0.2348, -1.0021,  0.5520, -0.8690, -1.1324, -0.7548, -0.0851],
            [-1.1453,  0.3631,  0.7027, -0.2346, -1.0011,  0.5515, -0.8682, -1.1313, -0.7540, -0.0850],
            [-1.1252,  0.3567,  0.6904, -0.2305, -0.9835,  0.5418, -0.8529, -1.1114, -0.7408, -0.0835],
            [-1.1239,  0.3563,  0.6896, -0.2302, -0.9824,  0.5412, -0.8519, -1.1101, -0.7399, -0.0834],
            [-1.0803,  0.3425,  0.6628, -0.2213, -0.9443,  0.5202, -0.8189, -1.0671, -0.7112, -0.0802],
            [-1.0909,  0.3458,  0.6693, -0.2234, -0.9535,  0.5253, -0.8269, -1.0775, -0.7182, -0.0810],
            [-1.0319,  0.3271,  0.6331, -0.2113, -0.9020,  0.4969, -0.7822, -1.0193, -0.6794, -0.0766],
            [-1.0757,  0.3410,  0.6600, -0.2203, -0.9402,  0.5179, -0.8153, -1.0625, -0.7082, -0.0799],
            [-1.0303,  0.3266,  0.6321, -0.2110, -0.9006,  0.4961, -0.7810, -1.0177, -0.6783, -0.0765],
            [-1.0913,  0.3459,  0.6696, -0.2235, -0.9539,  0.5255, -0.8272, -1.0779, -0.7185, -0.0810],
            [-1.0699,  0.3392,  0.6564, -0.2191, -0.9352,  0.5152, -0.8110, -1.0568, -0.7044, -0.0794],
            [-1.1174,  0.3542,  0.6856, -0.2288, -0.9767,  0.5380, -0.8470, -1.1037, -0.7356, -0.0830],
            [-1.1115,  0.3523,  0.6820, -0.2276, -0.9716,  0.5352, -0.8425, -1.0979, -0.7318, -0.0825],
            [-1.1382,  0.3608,  0.6983, -0.2331, -0.9949,  0.5481, -0.8627, -1.1243, -0.7493, -0.0845],
            [-1.1421,  0.3621,  0.7008, -0.2339, -0.9984,  0.5500, -0.8657, -1.1281, -0.7519, -0.0848],
        ]
        
        var output = latent
        for (index, t) in scheduler.timeSteps.enumerated() {
            latent = scheduler.scaleModelInput(timeStep: t, sample: latent)
            output = scheduler.step(output: output, timeStep: t, sample: latent, generator: generator)
            
            zip(output[0][0][0][0..<10].scalars, expectedOutputs[index]).forEach { actual, expected in
                XCTAssertEqual(actual, expected, accuracy: 0.02)
            }
        }
    }
}
