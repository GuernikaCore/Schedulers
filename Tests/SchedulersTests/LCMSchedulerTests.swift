//
//  LCMSchedulerTests.swift
//  
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import XCTest
import CoreML
import RandomGenerator
@testable import Schedulers

final class LCMSchedulerTests: XCTestCase {
    func test1StepsLinspace() throws {
        var scheduler = LCMScheduler(stepCount: 1, originalStepCount: 50)
        XCTAssertEqual(scheduler.timeSteps, [999])
        scheduler = LCMScheduler(strength: 0.9, stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [899])
    }
    
    func test2StepsLinspace() throws {
        let scheduler = LCMScheduler(stepCount: 2, originalStepCount: 50)
        XCTAssertEqual(scheduler.timeSteps, [999, 499])
    }
    
    func test33StepsLinspace() throws {
        let scheduler = LCMScheduler(stepCount: 33, originalStepCount: 50, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 979, 939, 919, 879, 859, 819, 799, 759, 739, 699, 679, 639, 619, 579, 559, 519, 499, 459, 439, 399, 379,
            339, 319, 279, 259, 219, 199, 159, 139,  99,  79,  39
        ])
    }
    
    func test33StepsLeading() throws {
        let scheduler = LCMScheduler(stepCount: 33, originalStepCount: 50, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 979, 939, 919, 879, 859, 819, 799, 759, 739, 699, 679, 639, 619, 579, 559, 519, 499, 459, 439, 399, 379,
            339, 319, 279, 259, 219, 199, 159, 139,  99,  79,  39
        ])
    }
    
    func test33StepsTrailing() throws {
        let scheduler = LCMScheduler(stepCount: 33, originalStepCount: 50, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            999, 979, 939, 919, 879, 859, 819, 799, 759, 739, 699, 679, 639, 619, 579, 559, 519, 499, 459, 439, 399, 379,
            339, 319, 279, 259, 219, 199, 159, 139,  99,  79,  39
        ])
    }
    func test50StepsLinspace() throws {
        var scheduler = LCMScheduler(stepCount: 50, originalStepCount: 70, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            979, 965, 951, 923, 909, 881, 867, 853, 825, 811, 783, 769, 755, 727, 713, 685, 671, 657, 629, 615, 587, 573,
            559, 531, 517, 489, 475, 461, 433, 419, 391, 377, 363, 335, 321, 293, 279, 265, 237, 223, 195, 181, 167, 139,
            125, 111,  83,  69,  41,  27
        ])
        scheduler = LCMScheduler(strength: 0.75, stepCount: 50, originalStepCount: 70, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [
            727, 713, 699, 685, 671, 657, 643, 629, 615, 601, 587, 573, 559, 545, 531, 517, 503, 489, 475, 461, 447, 433,
            419, 405, 391, 363, 349, 335, 321, 307, 293, 279, 265, 251, 237, 223, 209, 195, 181, 167, 153, 139, 125, 111,
            97,  83,  69,  55,  41,  27
        ])
    }
    
    func test50StepsLeading() throws {
        var scheduler = LCMScheduler(stepCount: 50, originalStepCount: 70, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            979, 965, 951, 923, 909, 881, 867, 853, 825, 811, 783, 769, 755, 727, 713, 685, 671, 657, 629, 615, 587, 573,
            559, 531, 517, 489, 475, 461, 433, 419, 391, 377, 363, 335, 321, 293, 279, 265, 237, 223, 195, 181, 167, 139,
            125, 111,  83,  69,  41,  27
        ])
        scheduler = LCMScheduler(strength: 0.75, stepCount: 50, originalStepCount: 70, timestepSpacing: .leading)
        XCTAssertEqual(scheduler.timeSteps, [
            727, 713, 699, 685, 671, 657, 643, 629, 615, 601, 587, 573, 559, 545, 531, 517, 503, 489, 475, 461, 447, 433,
            419, 405, 391, 363, 349, 335, 321, 307, 293, 279, 265, 251, 237, 223, 209, 195, 181, 167, 153, 139, 125, 111,
            97,  83,  69,  55,  41,  27
        ])
    }
    
    func test50StepsTrailing() throws {
        var scheduler = LCMScheduler(stepCount: 50, originalStepCount: 70, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            979, 965, 951, 923, 909, 881, 867, 853, 825, 811, 783, 769, 755, 727, 713, 685, 671, 657, 629, 615, 587, 573,
            559, 531, 517, 489, 475, 461, 433, 419, 391, 377, 363, 335, 321, 293, 279, 265, 237, 223, 195, 181, 167, 139,
            125, 111,  83,  69,  41,  27
        ])
        scheduler = LCMScheduler(strength: 0.75, stepCount: 50, originalStepCount: 70, timestepSpacing: .trailing)
        XCTAssertEqual(scheduler.timeSteps, [
            727, 713, 699, 685, 671, 657, 643, 629, 615, 601, 587, 573, 559, 545, 531, 517, 503, 489, 475, 461, 447, 433,
            419, 405, 391, 363, 349, 335, 321, 307, 293, 279, 265, 251, 237, 223, 209, 195, 181, 167, 153, 139, 125, 111,
            97,  83,  69,  55,  41,  27
        ])
    }
    
    func testStepEpsilon() throws {
        let scheduler = LCMScheduler(stepCount: 20)
        let generator: RandomGenerator = TorchRandomGenerator(seed: 50)
        let stdev = scheduler.initNoiseSigma
        var latent: MLShapedArray<Float32> = generator.nextArray(shape: [1, 4, 64, 64], mean: 0, stdev: stdev)
        zip(latent[0][0][0][0..<10].scalars, [
            -1.1588,  0.3673,  0.7110, -0.2373, -1.0129,  0.5580, -0.8784, -1.1446, -0.7629, -0.0860
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        let expectedOutputs: [[Float32]] = [
            [ 0.2347,  0.0228, -0.3498,  0.2134, -0.6773, -0.5356,  0.1405, -0.4654,  0.7950, -0.5241],
            [-0.0771, -0.0032,  2.5217,  0.0571,  1.9976,  1.2100, -0.5727, -2.5038, -1.6320,  1.5371],
            [-2.8436, -0.5624, -1.6426,  0.1175, -3.6636,  0.1564, -1.2410,  1.6719,  0.8225, -0.3700],
            [ 2.5582,  0.6167,  2.7536, -1.1972,  2.0579,  0.7186,  0.0495, -3.5615, -3.0227, -1.1124],
            [-6.3725,  0.0331, -2.4926,  1.6477, -2.1229, -0.7550, -1.4193,  1.5409,  3.3027,  0.8313],
            [ 6.8531, -1.0056,  4.4800, -1.5567,  3.2472, -0.0636,  1.1185, -4.0332, -4.8929, -2.9281],
            [-7.6018,  1.4150, -3.6733,  1.7014, -3.8108,  0.8917, -1.6773,  3.4299,  6.8453,  4.0531],
            [ 7.3323, -0.3070,  5.9628, -2.4651,  4.5425, -0.4365,  0.0303, -6.8822, -8.4790, -4.9238],
            [    -8.9754,     -0.0012,     -6.5146,      2.5040,     -5.8450,      1.2239,     -2.2431,      5.8201,
                  8.3838,      3.9871],
            [  8.6565,  -0.5591,   7.5195,  -2.7446,   4.7528,  -0.1420,   3.0902,  -8.7882, -10.1562,  -2.7398],
            [-10.1297,   0.1632,  -6.2123,   2.2859,  -3.3815,  -0.7461,  -4.1110,   5.6926,   8.1550,   2.6255],
            [ 8.1328, -1.2791,  6.9251, -2.8272,  2.0875,  1.2397,  2.8776, -7.5768, -7.3300, -2.3267],
            [-7.5426,  1.4230, -4.3181,  2.8433, -2.2923, -0.1955, -2.6982,  5.4263,  4.6761,  1.2592],
            [ 5.3679, -1.4588,  3.4676, -2.5760,  0.6266,  0.7474,  2.1412, -6.1366, -5.1987, -0.8433],
            [-5.7290,  1.4289, -1.7925,  1.5858, -2.9161,  0.0624, -1.9863,  3.8756,  4.2408, -0.0067],
            [ 2.5530,  0.7950,  2.4264, -1.6891,  1.0547,  0.4199, -0.0382, -2.8012, -3.8177,  0.7944],
            [-2.5509, -0.2792, -0.3983,  1.0442, -1.3717,  0.2236, -1.0801, -0.0029,  2.0687, -0.5577],
            [ 0.1129,  0.3533,  0.6516, -0.7654, -0.2594,  1.1623, -0.4562, -1.0384, -1.7727,  0.3105],
            [-1.7029,  0.5549,  0.3802,  0.3886, -1.0277,  0.2869, -0.3462, -0.8970, -0.0873, -0.4582],
            [-0.7713,  0.2407,  0.6382, -0.3410, -0.7885,  0.5037, -0.8191, -0.9567, -0.7644,  0.0251],
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
