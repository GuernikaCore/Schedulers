//
//  PNDMSchedulerTests.swift
//  
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import XCTest
import CoreML
import RandomGenerator
@testable import Schedulers

final class PNDMSchedulerTests: XCTestCase {
    func test1StepsLinspace() throws {
        var scheduler = PNDMScheduler(stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [0])
        scheduler = PNDMScheduler(strength: 0.9, stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [0])
    }
    
    func test2StepsLinspace() throws {
        let scheduler = PNDMScheduler(stepCount: 2)
        XCTAssertEqual(scheduler.timeSteps, [500,   0,   0])
    }
    
    func test33StepsLinspace() throws {
        let scheduler = PNDMScheduler(stepCount: 33, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 968, 968, 937, 905, 874, 843, 812, 780, 749, 718, 687, 656, 624, 593, 562, 531, 500, 468, 437, 406, 375,
            343, 312, 281, 250, 219, 187, 156, 125,  94,  62,  31,   0
        ])
    }
    
    func test33StepsLeading() throws {
        let scheduler = PNDMScheduler(stepCount: 33, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            960, 930, 930, 900, 870, 840, 810, 780, 750, 720, 690, 660, 630, 600, 570, 540, 510, 480, 450, 420, 390, 360,
            330, 300, 270, 240, 210, 180, 150, 120,  90,  60,  30,   0
        ])
    }
    
    func test33StepsTrailing() throws {
        let scheduler = PNDMScheduler(stepCount: 33, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 969, 969, 938, 908, 878, 847, 817, 787, 757, 726, 696, 666, 635, 605, 575, 544, 514, 484, 454, 423, 393,
            363, 332, 302, 272, 241, 211, 181, 151, 120,  90,  60,  29
        ])
    }
    func test50StepsLinspace() throws {
        var scheduler = PNDMScheduler(stepCount: 50, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 979, 979, 958, 938, 917, 897, 877, 856, 836, 816, 795, 775, 754, 734, 714, 693, 673, 652, 632, 612, 591,
            571, 550, 530, 510, 489, 469, 449, 428, 408, 387, 367, 347, 326, 306, 285, 265, 245, 224, 204, 183, 163, 143,
            122, 102,  82,  61,  41,  20,   0
        ])
        scheduler = PNDMScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            714, 693, 673, 652, 632, 612, 591, 571, 550, 530, 510, 489, 469, 449, 428, 408, 387, 367, 347, 326, 306, 285,
            265, 245, 224, 204, 183, 163, 143, 122, 102,  82,  61,  41,  20,   0
        ])
    }
    
    func test50StepsLeading() throws {
        var scheduler = PNDMScheduler(stepCount: 50, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            980, 960, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740, 720, 700, 680, 660, 640, 620, 600, 580,
            560, 540, 520, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160, 140,
            120, 100,  80,  60,  40,  20,   0
        ])
        scheduler = PNDMScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            700, 680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300, 280,
            260, 240, 220, 200, 180, 160, 140, 120, 100,  80,  60,  40,  20,   0
        ])
    }
    
    func test50StepsTrailing() throws {
        var scheduler = PNDMScheduler(stepCount: 50, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 979, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659, 639, 619, 599,
            579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 379, 359, 339, 319, 299, 279, 259, 239, 219, 199, 179, 159,
            139, 119,  99,  79,  59,  39,  19
        ])
        scheduler = PNDMScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            719, 699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 379, 359, 339, 319, 299,
            279, 259, 239, 219, 199, 179, 159, 139, 119,  99,  79,  59,  39,  19
        ])
    }
    
    func testStepEpsilon() throws {
        let scheduler = PNDMScheduler(stepCount: 20)
        let generator: RandomGenerator = TorchRandomGenerator(seed: 50)
        let stdev = scheduler.initNoiseSigma
        var latent: MLShapedArray<Float32> = generator.nextArray(shape: [1, 4, 64, 64], mean: 0, stdev: stdev)
        zip(latent[0][0][0][0..<10].scalars, [
            -1.1588,  0.3673,  0.7110, -0.2373, -1.0129,  0.5580, -0.8784, -1.1446, -0.7629, -0.0860
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        let expectedOutputs: [[Float32]] = [
            [-1.1568,  0.3667,  0.7098, -0.2369, -1.0112,  0.5570, -0.8769, -1.1427, -0.7616, -0.0859],
            [-1.1571,  0.3668,  0.7100, -0.2370, -1.0115,  0.5572, -0.8771, -1.1430, -0.7618, -0.0859],
            [-1.1565,  0.3666,  0.7096, -0.2369, -1.0109,  0.5569, -0.8766, -1.1423, -0.7614, -0.0859],
            [-1.1549,  0.3661,  0.7086, -0.2365, -1.0095,  0.5561, -0.8754, -1.1407, -0.7603, -0.0857],
            [-1.1538,  0.3657,  0.7079, -0.2363, -1.0085,  0.5556, -0.8746, -1.1396, -0.7596, -0.0857],
            [-1.1510,  0.3649,  0.7062, -0.2357, -1.0061,  0.5542, -0.8725, -1.1369, -0.7578, -0.0855],
            [-1.1491,  0.3643,  0.7051, -0.2354, -1.0045,  0.5533, -0.8710, -1.1351, -0.7565, -0.0853],
            [-1.1452,  0.3630,  0.7027, -0.2346, -1.0011,  0.5515, -0.8681, -1.1312, -0.7540, -0.0850],
            [-1.1425,  0.3622,  0.7010, -0.2340, -0.9987,  0.5501, -0.8660, -1.1285, -0.7522, -0.0848],
            [-1.1376,  0.3606,  0.6980, -0.2330, -0.9944,  0.5478, -0.8623, -1.1237, -0.7490, -0.0845],
            [-1.1340,  0.3595,  0.6958, -0.2323, -0.9912,  0.5460, -0.8596, -1.1201, -0.7466, -0.0842],
            [-1.1284,  0.3577,  0.6923, -0.2311, -0.9864,  0.5433, -0.8553, -1.1146, -0.7429, -0.0838],
            [-1.1240,  0.3563,  0.6896, -0.2302, -0.9825,  0.5412, -0.8520, -1.1102, -0.7400, -0.0835],
            [-1.1179,  0.3544,  0.6859, -0.2290, -0.9772,  0.5383, -0.8474, -1.1042, -0.7360, -0.0830],
            [-1.1127,  0.3527,  0.6827, -0.2279, -0.9726,  0.5358, -0.8434, -1.0990, -0.7325, -0.0826],
            [-1.1058,  0.3505,  0.6785, -0.2265, -0.9666,  0.5325, -0.8382, -1.0923, -0.7280, -0.0821],
            [-1.0987,  0.3483,  0.6741, -0.2250, -0.9603,  0.5290, -0.8328, -1.0852, -0.7233, -0.0816],
            [-1.0882,  0.3450,  0.6677, -0.2229, -0.9512,  0.5240, -0.8248, -1.0749, -0.7164, -0.0808],
            [-1.0706,  0.3394,  0.6569, -0.2193, -0.9358,  0.5155, -0.8115, -1.0575, -0.7049, -0.0795],
            [-0.9798,  0.3106,  0.6011, -0.2007, -0.8564,  0.4718, -0.7427, -0.9678, -0.6451, -0.0727],
            [-1.1588,  0.3673,  0.7110, -0.2373, -1.0129,  0.5580, -0.8784, -1.1446, -0.7629, -0.0860],
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
