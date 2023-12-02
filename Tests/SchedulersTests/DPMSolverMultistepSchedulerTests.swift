//
//  DPMSolverMultistepSchedulerTests.swift
//
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import XCTest
import CoreML
import RandomGenerator
@testable import Schedulers

final class DPMSolverMultistepSchedulerTests: XCTestCase {
    func test1StepsLinspace() throws {
        var scheduler = DPMSolverMultistepScheduler(stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [999])
        scheduler = DPMSolverMultistepScheduler(strength: 0.9, stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [999])
    }
    
    func test2StepsLinspace() throws {
        let scheduler = DPMSolverMultistepScheduler(stepCount: 2)
        XCTAssertEqual(scheduler.timeSteps, [999,   500])
    }
    
    func test33StepsLinspace() throws {
        let scheduler = DPMSolverMultistepScheduler(stepCount: 33, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 969, 938, 908, 878, 848, 817, 787, 757, 727, 696, 666, 636, 605, 575, 545, 515, 484, 454, 424, 394, 363,
            333, 303, 272, 242, 212, 182, 151, 121,  91,  61,  30
        ])
    }
    
    func test33StepsLeading() throws {
        let scheduler = DPMSolverMultistepScheduler(stepCount: 33, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            957, 928, 899, 870, 841, 812, 783, 754, 725, 696, 667, 638, 609, 580, 551, 522, 493, 464, 435, 406, 377, 348,
            319, 290, 261, 232, 203, 174, 145, 116,  87,  58,  29
        ])
    }
    
    func test33StepsTrailing() throws {
        let scheduler = DPMSolverMultistepScheduler(stepCount: 33, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 969, 938, 908, 878, 847, 817, 787, 757, 726, 696, 666, 635, 605, 575, 544, 514, 484, 454, 423, 393, 363,
            332, 302, 272, 241, 211, 181, 151, 120,  90,  60,  29
        ])
    }
    
    func test33StepsKarras() throws {
        let scheduler = DPMSolverMultistepScheduler(stepCount: 33, timestepSpacing: .linspace, useKarrasSigmas: true)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 977, 954, 930, 905, 878, 849, 819, 786, 751, 714, 673, 630, 584, 535, 482, 427, 371, 315, 259, 207, 160,
            119,  86,  59,  39,  25,  15,   9,   5,   2,   1,   0
        ])
    }
    
    func test50StepsLinspace() throws {
        var scheduler = DPMSolverMultistepScheduler(stepCount: 50, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659, 639, 619, 599, 579,
            559, 539, 519, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160, 140,
            120, 100,  80,  60,  40,  20
        ])
        scheduler = DPMSolverMultistepScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300, 280,
            260, 240, 220, 200, 180, 160, 140, 120, 100,  80,  60,  40,  20
        ])
    }
    
    func test50StepsLeading() throws {
        var scheduler = DPMSolverMultistepScheduler(stepCount: 50, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            950, 931, 912, 893, 874, 855, 836, 817, 798, 779, 760, 741, 722, 703, 684, 665, 646, 627, 608, 589, 570, 551,
            532, 513, 494, 475, 456, 437, 418, 399, 380, 361, 342, 323, 304, 285, 266, 247, 228, 209, 190, 171, 152, 133,
            114,  95,  76,  57,  38,  19
        ])
        scheduler = DPMSolverMultistepScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            665, 646, 627, 608, 589, 570, 551, 532, 513, 494, 475, 456, 437, 418, 399, 380, 361, 342, 323, 304, 285, 266,
            247, 228, 209, 190, 171, 152, 133, 114,  95,  76,  57,  38,  19
        ])
    }
    
    func test50StepsTrailing() throws {
        var scheduler = DPMSolverMultistepScheduler(stepCount: 50, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659, 639, 619, 599, 579,
            559, 539, 519, 499, 479, 459, 439, 419, 399, 379, 359, 339, 319, 299, 279, 259, 239, 219, 199, 179, 159, 139,
            119,  99,  79,  59,  39,  19
        ])
        scheduler = DPMSolverMultistepScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 379, 359, 339, 319, 299, 279,
            259, 239, 219, 199, 179, 159, 139, 119,  99,  79,  59,  39,  19
        ])
    }
    
    func test50StepsKarras() throws {
        var scheduler = DPMSolverMultistepScheduler(stepCount: 50, timestepSpacing: .linspace, useKarrasSigmas: true)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 985, 970, 955, 940, 924, 907, 890, 871, 853, 833, 813, 791, 769, 746, 721, 696, 669, 641, 612, 581, 549,
            516, 481, 446, 409, 372, 335, 299, 263, 228, 195, 165, 137, 112,  90,  71,  55,  42,  32,  23,  17,  12,   8,
            6,   4,   2,   1,   1,   0
        ])
        scheduler = DPMSolverMultistepScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .linspace, useKarrasSigmas: true)
        XCTAssertEqual(scheduler.timeSteps, [
            721, 696, 669, 641, 612, 581, 549, 516, 481, 446, 409, 372, 335, 299, 263, 228, 195, 165, 137, 112,  90,  71,
            55,  42,  32,  23,  17,  12,   8,   6,   4,   2,   1,   1,   0
        ])
    }
    
    func testStepEpsilon() throws {
        let scheduler = DPMSolverMultistepScheduler(stepCount: 20)
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
            [-1.1575,  0.3669,  0.7102, -0.2371, -1.0118,  0.5574, -0.8774, -1.1434, -0.7621, -0.0859],
            [-1.1563,  0.3665,  0.7095, -0.2368, -1.0107,  0.5568, -0.8765, -1.1421, -0.7613, -0.0859],
            [-1.1554,  0.3663,  0.7089, -0.2366, -1.0099,  0.5563, -0.8758, -1.1412, -0.7607, -0.0858],
            [-1.1537,  0.3657,  0.7079, -0.2363, -1.0085,  0.5555, -0.8745, -1.1396, -0.7595, -0.0857],
            [-1.1518,  0.3651,  0.7067, -0.2359, -1.0068,  0.5546, -0.8731, -1.1377, -0.7583, -0.0855],
            [-1.1493,  0.3643,  0.7052, -0.2354, -1.0046,  0.5534, -0.8712, -1.1352, -0.7566, -0.0853],
            [-1.1464,  0.3634,  0.7034, -0.2348, -1.0021,  0.5520, -0.8690, -1.1323, -0.7547, -0.0851],
            [-1.1430,  0.3623,  0.7013, -0.2341, -0.9991,  0.5504, -0.8664, -1.1290, -0.7525, -0.0849],
            [-1.1396,  0.3613,  0.6992, -0.2334, -0.9962,  0.5488, -0.8638, -1.1257, -0.7503, -0.0846],
            [-1.1351,  0.3598,  0.6964, -0.2325, -0.9922,  0.5465, -0.8604, -1.1212, -0.7473, -0.0843],
            [-1.1307,  0.3584,  0.6938, -0.2316, -0.9884,  0.5445, -0.8571, -1.1169, -0.7444, -0.0840],
            [-1.1259,  0.3569,  0.6908, -0.2306, -0.9842,  0.5422, -0.8535, -1.1122, -0.7413, -0.0836],
            [-1.1210,  0.3554,  0.6878, -0.2296, -0.9799,  0.5398, -0.8497, -1.1073, -0.7381, -0.0832],
            [-1.1159,  0.3537,  0.6847, -0.2285, -0.9754,  0.5373, -0.8458, -1.1022, -0.7347, -0.0829],
            [-1.1105,  0.3520,  0.6813, -0.2274, -0.9707,  0.5347, -0.8417, -1.0969, -0.7311, -0.0824],
            [-1.1045,  0.3501,  0.6777, -0.2262, -0.9654,  0.5318, -0.8372, -1.0910, -0.7271, -0.0820],
            [-1.0974,  0.3479,  0.6733, -0.2248, -0.9592,  0.5284, -0.8318, -1.0840, -0.7225, -0.0815],
            [-1.0883,  0.3450,  0.6677, -0.2229, -0.9513,  0.5240, -0.8249, -1.0749, -0.7165, -0.0808],
            [-1.1731,  0.3719,  0.7197, -0.2403, -1.0254,  0.5648, -0.8892, -1.1587, -0.7723, -0.0871],
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
        let scheduler = DPMSolverMultistepScheduler(stepCount: 20, useKarrasSigmas: true)
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
            [-1.1579,  0.3670,  0.7104, -0.2371, -1.0121,  0.5575, -0.8777, -1.1437, -0.7623, -0.0860],
            [-1.1570,  0.3668,  0.7099, -0.2370, -1.0113,  0.5571, -0.8770, -1.1428, -0.7617, -0.0859],
            [-1.1560,  0.3665,  0.7093, -0.2368, -1.0105,  0.5567, -0.8763, -1.1419, -0.7611, -0.0858],
            [-1.1541,  0.3658,  0.7081, -0.2364, -1.0088,  0.5557, -0.8748, -1.1400, -0.7598, -0.0857],
            [-1.1511,  0.3649,  0.7062, -0.2357, -1.0061,  0.5542, -0.8725, -1.1370, -0.7578, -0.0855],
            [-1.1458,  0.3632,  0.7030, -0.2347, -1.0016,  0.5517, -0.8685, -1.1318, -0.7544, -0.0851],
            [-1.1376,  0.3606,  0.6980, -0.2330, -0.9944,  0.5478, -0.8623, -1.1237, -0.7490, -0.0845],
            [-1.1256,  0.3568,  0.6906, -0.2305, -0.9839,  0.5420, -0.8532, -1.1118, -0.7410, -0.0836],
            [-1.1104,  0.3520,  0.6813, -0.2274, -0.9706,  0.5347, -0.8417, -1.0968, -0.7310, -0.0824],
            [-1.0951,  0.3471,  0.6719, -0.2243, -0.9572,  0.5273, -0.8301, -1.0817, -0.7210, -0.0813],
            [-1.0848,  0.3439,  0.6656, -0.2222, -0.9482,  0.5224, -0.8223, -1.0715, -0.7142, -0.0805],
            [-1.0829,  0.3433,  0.6644, -0.2218, -0.9466,  0.5215, -0.8209, -1.0697, -0.7130, -0.0804],
            [-1.0893,  0.3453,  0.6683, -0.2231, -0.9521,  0.5245, -0.8257, -1.0759, -0.7171, -0.0809],
            [-1.1008,  0.3489,  0.6754, -0.2254, -0.9622,  0.5300, -0.8344, -1.0873, -0.7247, -0.0817],
            [-1.1139,  0.3531,  0.6834, -0.2281, -0.9737,  0.5364, -0.8443, -1.1003, -0.7334, -0.0827],
            [-1.1262,  0.3570,  0.6910, -0.2307, -0.9844,  0.5423, -0.8537, -1.1124, -0.7415, -0.0836],
            [-1.1364,  0.3602,  0.6973, -0.2327, -0.9934,  0.5472, -0.8614, -1.1225, -0.7482, -0.0844],
            [-1.1442,  0.3627,  0.7020, -0.2343, -1.0002,  0.5510, -0.8673, -1.1302, -0.7533, -0.0850],
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
