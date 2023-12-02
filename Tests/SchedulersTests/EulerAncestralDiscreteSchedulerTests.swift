//
//  EulerDiscreteSchedulerTests.swift
//  
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import XCTest
import CoreML
import RandomGenerator
@testable import Schedulers

final class EulerAncestralDiscreteSchedulerTests: XCTestCase {
    func test1StepsLinspace() throws {
        var scheduler = EulerAncestralDiscreteScheduler(stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [0])
        scheduler = EulerAncestralDiscreteScheduler(strength: 0.9, stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [0])
    }
    
    func test2StepsLinspace() throws {
        let scheduler = EulerAncestralDiscreteScheduler(stepCount: 2)
        XCTAssertEqual(scheduler.timeSteps, [999,   0])
    }
    
    func test33StepsLinspace() throws {
        let scheduler = EulerAncestralDiscreteScheduler(stepCount: 33, timestepSpacing: .linspace)
        print(scheduler.sigmas)
        zip(scheduler.timeSteps, [
            999.00000000, 967.78125000, 936.56250000, 905.34375000, 874.12500000, 842.90625000, 811.68750000,
            780.46875000, 749.25000000, 718.03125000, 686.81250000, 655.59375000, 624.37500000, 593.15625000,
            561.93750000, 530.71875000, 499.50000000, 468.28125000, 437.06250000, 405.84375000, 374.62500000,
            343.40625000, 312.18750000, 280.96875000, 249.75000000, 218.53125000, 187.31250000, 156.09375000,
            124.87500000,  93.65625000,  62.43750000,  31.21875000,   0.00000000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test33StepsLeading() throws {
        let scheduler = EulerAncestralDiscreteScheduler(stepCount: 33, timestepSpacing: .leading)
        zip(scheduler.timeSteps, [
            960, 930, 900, 870, 840, 810, 780, 750, 720, 690, 660, 630, 600, 570, 540, 510, 480, 450,
            420, 390, 360, 330, 300, 270, 240, 210, 180, 150, 120,  90,  60,  30,   0
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test33StepsTrailing() throws {
        let scheduler = EulerAncestralDiscreteScheduler(stepCount: 33, timestepSpacing: .trailing)
        zip(scheduler.timeSteps, [
            999, 969, 938, 908, 878, 847, 817, 787, 757, 726, 696, 666, 635, 605, 575, 544, 514, 484,
            454, 423, 393, 363, 332, 302, 272, 241, 211, 181, 151, 120,  90,  60,  29
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    func test50StepsLinspace() throws {
        var scheduler = EulerAncestralDiscreteScheduler(stepCount: 50, timestepSpacing: .linspace)
        zip(scheduler.timeSteps, [
            999.0000000000000000, 978.6122436523437500, 958.2244873046875000, 937.8367309570312500, 917.4489746093750000,
            897.0612182617187500, 876.6734619140625000, 856.2857055664062500, 835.8979492187500000, 815.5101928710937500,
            795.1224365234375000, 774.7346801757812500, 754.3469238281250000, 733.9591674804687500, 713.5714111328125000,
            693.1836547851562500, 672.7958984375000000, 652.4081420898437500, 632.0203857421875000, 611.6326293945312500,
            591.2448730468750000, 570.8571166992187500, 550.4693603515625000, 530.0816040039062500, 509.6938781738281250,
            489.3061218261718750, 468.9183654785156250, 448.5306091308593750, 428.1428527832031250, 407.7550964355468750,
            387.3673400878906250, 366.9795837402343750, 346.5918273925781250, 326.2040710449218750, 305.8163146972656250,
            285.4285583496093750, 265.0408020019531250, 244.6530609130859375, 224.2653045654296875, 203.8775482177734375,
            183.4897918701171875, 163.1020355224609375, 142.7142791748046875, 122.3265304565429688, 101.9387741088867188,
            81.5510177612304688,  61.1632652282714844,  40.7755088806152344,  20.3877544403076172,   0.0000000000000000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        scheduler = EulerAncestralDiscreteScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .linspace)
        zip(scheduler.timeSteps, [
            693.1836547851562500, 672.7958984375000000, 652.4081420898437500, 632.0203857421875000, 611.6326293945312500,
            591.2448730468750000, 570.8571166992187500, 550.4693603515625000, 530.0816040039062500, 509.6938781738281250,
            489.3061218261718750, 468.9183654785156250, 448.5306091308593750, 428.1428527832031250, 407.7550964355468750,
            387.3673400878906250, 366.9795837402343750, 346.5918273925781250, 326.2040710449218750, 305.8163146972656250,
            285.4285583496093750, 265.0408020019531250, 244.6530609130859375, 224.2653045654296875, 203.8775482177734375,
            183.4897918701171875, 163.1020355224609375, 142.7142791748046875, 122.3265304565429688, 101.9387741088867188,
            81.5510177612304688,  61.1632652282714844,  40.7755088806152344,  20.3877544403076172,   0.0000000000000000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test50StepsLeading() throws {
        var scheduler = EulerAncestralDiscreteScheduler(stepCount: 50, timestepSpacing: .leading)
        zip(scheduler.timeSteps, [
            980, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740, 720, 700, 680, 660, 640,
            620, 600, 580, 560, 540, 520, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300, 280,
            260, 240, 220, 200, 180, 160, 140, 120, 100,  80,  60,  40,  20,   0
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        scheduler = EulerAncestralDiscreteScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .leading)
        zip(scheduler.timeSteps, [
            680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 480, 460, 440, 420, 400, 380, 360, 340,
            320, 300, 280, 260, 240, 220, 200, 180, 160, 140, 120, 100,  80,  60,  40,  20,   0
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test50StepsTrailing() throws {
        var scheduler = EulerAncestralDiscreteScheduler(stepCount: 50, timestepSpacing: .trailing)
        zip(scheduler.timeSteps, [
            999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659,
            639, 619, 599, 579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 379, 359, 339, 319, 299,
            279, 259, 239, 219, 199, 179, 159, 139, 119,  99,  79,  59,  39,  19
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        scheduler = EulerAncestralDiscreteScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .trailing)
        zip(scheduler.timeSteps, [
            699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 379, 359,
            339, 319, 299, 279, 259, 239, 219, 199, 179, 159, 139, 119,  99,  79,  59,  39,  19
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func testStepEpsilon() throws {
        let scheduler = EulerAncestralDiscreteScheduler(stepCount: 20)
        let generator: RandomGenerator = TorchRandomGenerator(seed: 50)
        let stdev = scheduler.initNoiseSigma
        var latent: MLShapedArray<Float32> = generator.nextArray(shape: [1, 4, 64, 64], mean: 0, stdev: stdev)
        zip(latent[0][0][0][0..<10].scalars, [
            -16.9352,   5.3684,  10.3907,  -3.4685, -14.8032,   8.1545, -12.8368, -16.7278, -11.1495,  -1.2574
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        let expectedOutputs: [[Float32]] = [
            [114.2532, -35.5071, -71.6053,  24.6086,  93.4189, -58.1037,  86.3299, 107.7567,  79.9022,   4.5238],
            [-523.6859,  163.2592,  340.0889, -111.2833, -423.0911,  269.7515, -398.7107, -511.6601, -370.3802,  -16.1273],
            [ 1731.1584,  -545.6743, -1126.0033,   371.0868,  1403.5518,  -891.0825,  1319.2133,  1697.5066,  1227.7899,
              59.9317],
            [-4162.5684,  1310.5828,  2707.2126,  -894.7767, -3380.0520,  2143.7764, -3174.1951, -4082.3813, -2955.9426,
              -148.7111],
            [ 7398.6763, -2330.2456, -4815.3691,  1592.7401,  6015.2910, -3814.3157,  5644.8511,  7257.9355,  5259.1030,
              263.6326],
            [-9920.2764,  3121.9846,  6458.1426, -2134.3027, -8062.4854,  5111.5278, -7568.5020, -9733.8252, -7051.9204,
              -356.9722],
            [10224.1514, -3217.1667, -6653.9624,  2199.6504,  8309.2783, -5266.7896,  7799.6899, 10030.4883,  7270.1318,
             369.2727],
            [-8248.6240,  2596.6833,  5369.4883, -1774.9473, -6701.9614,  4249.0435, -6293.7373, -8094.4897, -5865.0845,
              -298.1678],
            [ 5307.7891, -1671.7463, -3456.5300,  1142.4281,  4312.6299, -2734.0732,  4048.6477,  5208.7969,  3774.7263,
              190.8060],
            [-2773.7861,   872.8928,  1806.6532,  -597.0192, -2254.3813,  1429.5714, -2114.5635, -2723.9814, -1973.5721,
              -98.2990],
            [1198.4784, -377.9865, -781.0200,  258.0873,  976.3688, -619.2455,  913.9232, 1176.5776,  852.8033,   42.6331],
            [-436.1480,  136.4254,  284.5697,  -94.2760, -355.2944,  225.3344, -332.5915, -429.0244, -309.6166,  -15.3742],
            [136.2692, -42.5656, -88.3989,  29.9452, 111.1648, -70.1484, 104.1448, 134.1623,  96.2688,   4.4320],
            [-36.6305,  11.1124,  23.6256,  -8.2097, -30.2997,  19.0861, -27.7210, -36.9277, -26.6548,  -1.0560],
            [ 8.3488, -2.6571, -5.8442,  2.0479,  6.5005, -4.6365,  6.9399,  9.5032,  7.3646, -0.0291],
            [-1.7167,  1.3392,  1.6050, -0.7095, -1.2488,  0.9413, -1.7272, -1.6016, -1.8266,  0.5198],
            [ 0.3356, -0.4145, -0.2716,  0.3774,  0.3796, -0.2711,  0.2546,  0.1926,  0.8685, -0.1402],
            [ 0.0281, -0.0012, -0.0932, -0.1019,  0.0226,  0.4242, -0.0642,  0.0478, -0.1991,  0.1098],
            [-0.0637,  0.0357,  0.0053,  0.0687, -0.0141, -0.0849,  0.0631, -0.0190,  0.0584, -0.0572],
            [ 0.0019, -0.0010, -0.0002, -0.0020,  0.0004,  0.0025, -0.0018,  0.0006, -0.0017,  0.0017],
        ]
        
        var output = latent
        for (index, t) in scheduler.timeSteps.enumerated() {
            latent = scheduler.scaleModelInput(timeStep: t, sample: latent)
            output = scheduler.step(output: output, timeStep: t, sample: latent, generator: generator)
            
            zip(output[0][0][0][0..<10].scalars, expectedOutputs[index]).forEach { actual, expected in
                XCTAssertEqual(actual, expected, accuracy: 0.05)
            }
        }
    }
}
