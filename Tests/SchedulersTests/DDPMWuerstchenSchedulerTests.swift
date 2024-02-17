//
//  DDPMWuerstchenSchedulerTests.swift
//
//
//  Created by Guillermo Cique Fernández on 17/02/24.
//

import XCTest
import CoreML
import RandomGenerator
@testable import Schedulers

final class DDPMWuerstchenSchedulerTests: XCTestCase {
    func test1Steps() throws {
        var scheduler = DDPMWuerstchenScheduler(stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [1])
        scheduler = DDPMWuerstchenScheduler(strength: 0.9, stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [1])
    }
    
    func test2Steps() throws {
        let scheduler = DDPMWuerstchenScheduler(stepCount: 2)
        XCTAssertEqual(scheduler.timeSteps, [1,   0.5])
    }
    
    func test33Steps() throws {
        let scheduler = DDPMWuerstchenScheduler(stepCount: 33)
        zip(scheduler.timeSteps, [
            1.0000, 0.9697, 0.9394, 0.9091, 0.8788, 0.8485, 0.8182, 0.7879, 0.7576, 0.7273, 0.6970, 0.6667, 0.6364, 0.6061,
            0.5758, 0.5455, 0.5152, 0.4848, 0.4545, 0.4242, 0.3939, 0.3636, 0.3333, 0.3030, 0.2727, 0.2424, 0.2121, 0.1818,
            0.1515, 0.1212, 0.0909, 0.0606, 0.0303
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test50Steps() throws {
        var scheduler = DDPMWuerstchenScheduler(stepCount: 50)
        zip(scheduler.timeSteps, [
            1.0000, 0.9800, 0.9600, 0.9400, 0.9200, 0.9000, 0.8800, 0.8600, 0.8400, 0.8200, 0.8000, 0.7800, 0.7600, 0.7400,
            0.7200, 0.7000, 0.6800, 0.6600, 0.6400, 0.6200, 0.6000, 0.5800, 0.5600, 0.5400, 0.5200, 0.5000, 0.4800, 0.4600,
            0.4400, 0.4200, 0.4000, 0.3800, 0.3600, 0.3400, 0.3200, 0.3000, 0.2800, 0.2600, 0.2400, 0.2200, 0.2000, 0.1800,
            0.1600, 0.1400, 0.1200, 0.1000, 0.0800, 0.0600, 0.0400, 0.0200
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        scheduler = DDPMWuerstchenScheduler(strength: 0.7, stepCount: 50)
        zip(scheduler.timeSteps, [
            0.7000, 0.6800, 0.6600, 0.6400, 0.6200, 0.6000, 0.5800, 0.5600, 0.5400, 0.5200, 0.5000, 0.4800, 0.4600, 0.4400,
            0.4200, 0.4000, 0.3800, 0.3600, 0.3400, 0.3200, 0.3000, 0.2800, 0.2600, 0.2400, 0.2200, 0.2000, 0.1800, 0.1600,
            0.1400, 0.1200, 0.1000, 0.0800, 0.0600, 0.0400, 0.0200
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func testStepEpsilon() throws {
        let scheduler = DDPMWuerstchenScheduler(stepCount: 20)
        let generator: RandomGenerator = TorchRandomGenerator(seed: 50)
        let stdev = scheduler.initNoiseSigma
        var latent: MLShapedArray<Float32> = generator.nextArray(shape: [1, 4, 64, 64], mean: 0, stdev: stdev)
        zip(latent[0][0][0][0..<10].scalars, [
            -1.1588,  0.3673,  0.7110, -0.2373, -1.0129,  0.5580, -0.8784, -1.1446, -0.7629, -0.0860
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        let expectedOutputs: [[Float32]] = [
            [ 0.0879,  0.0686, -0.2582,  0.1821, -0.7990, -0.4617,  0.0295, -0.6052,  0.6936, -0.5310],
            [-0.8428,  0.2147,  2.7142, -0.1582,  1.3057,  1.5428, -1.0718, -2.7237, -2.1084,  1.4287],
            [-2.1515, -0.3807, -0.7930,  0.1299, -2.5996,  0.2451, -1.0558,  0.5832,  0.4500, -0.1351],
            [-0.0314,  0.3154,  1.1899, -0.8676, -0.7286,  0.7173, -0.8146, -1.7703, -1.8887, -1.0032],
            [-2.6141,  0.4975,  0.2628,  0.4154, -0.0588,  0.0241, -0.9079, -1.3481,  0.3186,  0.1359],
            [-0.1392, -0.5239,  1.0286, -0.0200, -0.1398, -0.2664, -0.4426, -1.2348, -0.9549, -1.1281],
            [-0.6417,  0.5299,  0.7181, -0.1370, -0.6510,  0.8167, -0.5857, -0.7756,  0.5951,  0.7224],
            [-1.1027,  0.6781,  1.0313, -0.3402, -0.2867,  0.3574, -1.2173, -1.8231, -0.8961, -0.3862],
            [-1.0723, -0.0887, -0.0016, -0.0716, -1.0674,  0.5955, -1.2209, -0.7997, -0.3845, -0.4375],
            [-0.7366, -0.0095,  0.7319, -0.1811, -0.9315,  0.6836,  0.0888, -1.6700, -0.9749,  0.6620],
            [-1.4248,  0.0811,  0.6206, -0.2228, -0.0015, -0.1595, -1.0748, -1.4040, -0.8166, -0.1593],
            [-0.9206, -0.2575,  0.7994, -0.3950, -1.0313,  0.5920, -0.6976, -1.3744, -0.2245,  0.0175],
            [-0.9310,  0.4284,  0.8261,  0.1366, -0.6880,  0.5674, -0.5691, -0.7669, -0.9501, -0.3085],
            [-0.7614,  0.0137,  0.3282, -0.3070, -0.9655,  0.4807, -0.4265, -1.3199, -0.9173,  0.0564],
            [-1.4250,  0.4016,  0.6055, -0.1583, -1.4479,  0.4935, -0.6954, -0.6102, -0.0487, -0.2853],
            [-0.8942,  0.8284,  0.8555, -0.3955, -0.6994,  0.4454, -0.9334, -0.7524, -0.9164,  0.3026],
            [-1.0703,  0.1620,  0.6459, -0.0272, -0.8522,  0.4542, -0.8350, -1.1583, -0.3109, -0.1541],
            [-0.9661,  0.2915,  0.5316, -0.2551, -0.8448,  0.7646, -0.7890, -0.9465, -0.7490, -0.0098],
            [-1.2034,  0.4286,  0.6118, -0.0897, -0.9442,  0.5023, -0.6646, -1.0626, -0.6460, -0.1698],
            [-1.0567,  0.3308,  0.6596, -0.2303, -0.9332,  0.5157, -0.8229, -1.0549, -0.7087, -0.0713],
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
