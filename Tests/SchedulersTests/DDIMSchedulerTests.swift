//
//  DDIMSchedulerTests.swift
//
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import XCTest
import CoreML
import RandomGenerator
@testable import Schedulers

final class DDIMSchedulerTests: XCTestCase {
    func test1StepsLinspace() throws {
        var scheduler = DDIMScheduler(stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [0])
        scheduler = DDIMScheduler(strength: 0.9, stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [0])
    }
    
    func test2StepsLinspace() throws {
        let scheduler = DDIMScheduler(stepCount: 2)
        XCTAssertEqual(scheduler.timeSteps, [500,   0])
    }
    
    func test33StepsLinspace() throws {
        let scheduler = DDIMScheduler(stepCount: 33, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 968, 937, 905, 874, 843, 812, 780, 749, 718, 687, 656, 624, 593, 562, 531, 500, 468, 437, 406, 375, 343,
            312, 281, 250, 219, 187, 156, 125,  94,  62,  31,   0
        ])
    }
    
    func test33StepsLeading() throws {
        let scheduler = DDIMScheduler(stepCount: 33, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            960, 930, 900, 870, 840, 810, 780, 750, 720, 690, 660, 630, 600, 570, 540, 510, 480, 450, 420, 390, 360, 330,
            300, 270, 240, 210, 180, 150, 120,  90,  60,  30,   0
        ])
    }
    
    func test33StepsTrailing() throws {
        let scheduler = DDIMScheduler(stepCount: 33, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 969, 938, 908, 878, 847, 817, 787, 757, 726, 696, 666, 635, 605, 575, 544, 514, 484, 454, 423, 393, 363,
            332, 302, 272, 241, 211, 181, 151, 120,  90,  60,  29
        ])
    }
    func test50StepsLinspace() throws {
        var scheduler = DDIMScheduler(stepCount: 50, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 979, 958, 938, 917, 897, 877, 856, 836, 816, 795, 775, 754, 734, 714, 693, 673, 652, 632, 612, 591, 571,
            550, 530, 510, 489, 469, 449, 428, 408, 387, 367, 347, 326, 306, 285, 265, 245, 224, 204, 183, 163, 143, 122,
            102,  82,  61,  41,  20,   0
        ])
        scheduler = DDIMScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            693, 673, 652, 632, 612, 591, 571, 550, 530, 510, 489, 469, 449, 428, 408, 387, 367, 347, 326, 306, 285, 265,
            245, 224, 204, 183, 163, 143, 122, 102,  82,  61,  41,  20,   0
        ])
    }
    
    func test50StepsLeading() throws {
        var scheduler = DDIMScheduler(stepCount: 50, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            980, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740, 720, 700, 680, 660, 640, 620, 600, 580, 560,
            540, 520, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160, 140, 120,
            100,  80,  60,  40,  20,   0
        ])
        scheduler = DDIMScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300, 280, 260,
            240, 220, 200, 180, 160, 140, 120, 100,  80,  60,  40,  20,   0
        ])
    }
    
    func test50StepsTrailing() throws {
        var scheduler = DDIMScheduler(stepCount: 50, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659, 639, 619, 599, 579,
            559, 539, 519, 499, 479, 459, 439, 419, 399, 379, 359, 339, 319, 299, 279, 259, 239, 219, 199, 179, 159, 139,
            119,  99,  79,  59,  39,  19
        ])
        scheduler = DDIMScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 379, 359, 339, 319, 299, 279,
            259, 239, 219, 199, 179, 159, 139, 119,  99,  79,  59,  39,  19
        ])
    }
    
    func testStepEpsilon() throws {
        let scheduler = DDIMScheduler(stepCount: 20)
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
            [-1.1563,  0.3665,  0.7095, -0.2368, -1.0108,  0.5568, -0.8765, -1.1422, -0.7613, -0.0859],
            [-1.1549,  0.3661,  0.7086, -0.2365, -1.0096,  0.5561, -0.8754, -1.1408, -0.7604, -0.0858],
            [-1.1533,  0.3656,  0.7076, -0.2362, -1.0081,  0.5553, -0.8742, -1.1392, -0.7593, -0.0856],
            [-1.1512,  0.3649,  0.7063, -0.2358, -1.0062,  0.5543, -0.8726, -1.1371, -0.7579, -0.0855],
            [-1.1485,  0.3641,  0.7047, -0.2352, -1.0039,  0.5530, -0.8706, -1.1345, -0.7562, -0.0853],
            [-1.1454,  0.3631,  0.7028, -0.2346, -1.0012,  0.5515, -0.8682, -1.1314, -0.7541, -0.0850],
            [-1.1418,  0.3620,  0.7006, -0.2339, -0.9981,  0.5498, -0.8655, -1.1278, -0.7517, -0.0848],
            [-1.1378,  0.3607,  0.6981, -0.2330, -0.9945,  0.5479, -0.8624, -1.1238, -0.7491, -0.0845],
            [-1.1333,  0.3593,  0.6953, -0.2321, -0.9906,  0.5457, -0.8590, -1.1194, -0.7461, -0.0841],
            [-1.1285,  0.3577,  0.6924, -0.2311, -0.9864,  0.5434, -0.8554, -1.1147, -0.7430, -0.0838],
            [-1.1234,  0.3561,  0.6893, -0.2301, -0.9820,  0.5409, -0.8515, -1.1096, -0.7396, -0.0834],
            [-1.1180,  0.3544,  0.6859, -0.2290, -0.9772,  0.5383, -0.8474, -1.1043, -0.7360, -0.0830],
            [-1.1121,  0.3525,  0.6824, -0.2278, -0.9721,  0.5355, -0.8430, -1.0985, -0.7322, -0.0826],
            [-1.1057,  0.3505,  0.6784, -0.2265, -0.9665,  0.5324, -0.8381, -1.0922, -0.7280, -0.0821],
            [-1.0981,  0.3481,  0.6738, -0.2249, -0.9599,  0.5288, -0.8324, -1.0847, -0.7230, -0.0815],
            [-1.0879,  0.3449,  0.6675, -0.2228, -0.9510,  0.5238, -0.8246, -1.0746, -0.7162, -0.0808],
            [-1.0698,  0.3391,  0.6564, -0.2191, -0.9351,  0.5151, -0.8109, -1.0567, -0.7043, -0.0794],
            [-0.9774,  0.3098,  0.5997, -0.2002, -0.8543,  0.4706, -0.7408, -0.9654, -0.6435, -0.0726],
            [-1.0000,  0.3584,  0.6938, -0.2316, -0.9884,  0.5445, -0.8571, -1.0000, -0.7445, -0.0840],
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
