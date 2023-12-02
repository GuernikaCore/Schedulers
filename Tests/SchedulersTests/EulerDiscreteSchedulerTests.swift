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

final class EulerDiscreteSchedulerTests: XCTestCase {
    func test1StepsLinspace() throws {
        var scheduler = EulerDiscreteScheduler(stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [0])
        scheduler = EulerDiscreteScheduler(strength: 0.9, stepCount: 1)
        XCTAssertEqual(scheduler.timeSteps, [0])
    }
    
    func test2StepsLinspace() throws {
        let scheduler = EulerDiscreteScheduler(stepCount: 2)
        XCTAssertEqual(scheduler.timeSteps, [999,   0])
    }
    
    func test33StepsLinspace() throws {
        let scheduler = EulerDiscreteScheduler(stepCount: 33, timestepSpacing: .linspace)
        print(scheduler.sigmas)
        zip(scheduler.timeSteps, [
            999.00000000, 967.78125000, 936.56250000, 905.34375000, 874.12500000, 842.90625000, 811.68750000, 780.46875000,
            749.25000000, 718.03125000, 686.81250000, 655.59375000, 624.37500000, 593.15625000, 561.93750000, 530.71875000,
            499.50000000, 468.28125000, 437.06250000, 405.84375000, 374.62500000, 343.40625000, 312.18750000, 280.96875000,
            249.75000000, 218.53125000, 187.31250000, 156.09375000, 124.87500000,  93.65625000,  62.43750000,  31.21875000,
            0.00000000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test33StepsLeading() throws {
        let scheduler = EulerDiscreteScheduler(stepCount: 33, timestepSpacing: .leading)
        zip(scheduler.timeSteps, [
            960, 930, 900, 870, 840, 810, 780, 750, 720, 690, 660, 630, 600, 570, 540, 510, 480, 450,
            420, 390, 360, 330, 300, 270, 240, 210, 180, 150, 120,  90,  60,  30,   0
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test33StepsTrailing() throws {
        let scheduler = EulerDiscreteScheduler(stepCount: 33, timestepSpacing: .trailing)
        zip(scheduler.timeSteps, [
            999, 969, 938, 908, 878, 847, 817, 787, 757, 726, 696, 666, 635, 605, 575, 544, 514, 484,
            454, 423, 393, 363, 332, 302, 272, 241, 211, 181, 151, 120,  90,  60,  29
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test33StepsKarras() throws {
        let scheduler = EulerDiscreteScheduler(stepCount: 33, useKarrasSigmas: true)
        print(scheduler.timeSteps)
        print(scheduler.sigmas)
    }
    
    func test50StepsLinspace() throws {
        var scheduler = EulerDiscreteScheduler(stepCount: 50, timestepSpacing: .linspace)
        zip(scheduler.timeSteps, [
            999.00000000, 978.61224365, 958.22448730, 937.83673096, 917.44897461, 897.06121826, 876.67346191, 856.28570557,
            835.89794922, 815.51019287, 795.12243652, 774.73468018, 754.34692383, 733.95916748, 713.57141113, 693.18365479,
            672.79589844, 652.40814209, 632.02038574, 611.63262939, 591.24487305, 570.85711670, 550.46936035, 530.08160400,
            509.69387817, 489.30612183, 468.91836548, 448.53060913, 428.14285278, 407.75509644, 387.36734009, 366.97958374,
            346.59182739, 326.20407104, 305.81631470, 285.42855835, 265.04080200, 244.65306091, 224.26530457, 203.87754822,
            183.48979187, 163.10203552, 142.71427917, 122.32653046, 101.93877411,  81.55101776,  61.16326523,  40.77550888,
            20.38775444,   0.00000000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        scheduler = EulerDiscreteScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .linspace)
        zip(scheduler.timeSteps, [
            693.18365479, 672.79589844, 652.40814209, 632.02038574, 611.63262939, 591.24487305, 570.85711670, 550.46936035,
            530.08160400, 509.69387817, 489.30612183, 468.91836548, 448.53060913, 428.14285278, 407.75509644, 387.36734009,
            366.97958374, 346.59182739, 326.20407104, 305.81631470, 285.42855835, 265.04080200, 244.65306091, 224.26530457,
            203.87754822, 183.48979187, 163.10203552, 142.71427917, 122.32653046, 101.93877411,  81.55101776,  61.16326523,
            40.77550888,  20.38775444,   0.00000000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test50StepsLeading() throws {
        var scheduler = EulerDiscreteScheduler(stepCount: 50, timestepSpacing: .leading)
        zip(scheduler.timeSteps, [
            980, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740, 720, 700, 680, 660, 640,
            620, 600, 580, 560, 540, 520, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300, 280,
            260, 240, 220, 200, 180, 160, 140, 120, 100,  80,  60,  40,  20,   0
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        scheduler = EulerDiscreteScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .leading)
        zip(scheduler.timeSteps, [
            680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 480, 460, 440, 420, 400, 380, 360, 340,
            320, 300, 280, 260, 240, 220, 200, 180, 160, 140, 120, 100,  80,  60,  40,  20,   0
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test50StepsTrailing() throws {
        var scheduler = EulerDiscreteScheduler(stepCount: 50, timestepSpacing: .trailing)
        zip(scheduler.timeSteps, [
            999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659,
            639, 619, 599, 579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 379, 359, 339, 319, 299,
            279, 259, 239, 219, 199, 179, 159, 139, 119,  99,  79,  59,  39,  19
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        scheduler = EulerDiscreteScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .trailing)
        zip(scheduler.timeSteps, [
            699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 379, 359,
            339, 319, 299, 279, 259, 239, 219, 199, 179, 159, 139, 119,  99,  79,  59,  39,  19
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test50StepsKarras() throws {
        let scheduler = EulerDiscreteScheduler(stepCount: 50, useKarrasSigmas: true)
        print(scheduler.timeSteps)
        print(scheduler.sigmas)
    }
    
    func testStepEpsilon() throws {
        let scheduler = EulerDiscreteScheduler(stepCount: 20)
        let generator: RandomGenerator = TorchRandomGenerator(seed: 50)
        let stdev = scheduler.initNoiseSigma
        var latent: MLShapedArray<Float32> = generator.nextArray(shape: [1, 4, 64, 64], mean: 0, stdev: stdev)
        zip(latent[0][0][0][0..<10].scalars, [
            -16.9352,   5.3684,  10.3907,  -3.4685, -14.8032,   8.1545, -12.8368, -16.7278, -11.1495,  -1.2574
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        let expectedOutputs: [[Float32]] = [
            [ 64.3457, -20.3974, -39.4796,  13.1786,  56.2454, -30.9834,  48.7739,  63.5577,  42.3628,   4.7774],
            [-171.6086,   54.3993,  105.2913,  -35.1470, -150.0051,   82.6322, -130.0789, -169.5069, -112.9806,  -12.7413],
            [ 322.0250, -102.0808, -197.5800,   65.9537,  281.4859, -155.0599,  244.0941,  318.0812,  212.0091,   23.9092],
            [-434.5059,  137.7369,  266.5932,  -88.9908, -379.8068,  209.2212, -329.3544, -429.1846, -286.0623,  -32.2605],
            [ 430.2823, -136.3980, -264.0019,   88.1258,  376.1150, -207.1875,  326.1530,  425.0127,  283.2817,   31.9469],
            [-319.1042,  101.1549,  195.7879,  -65.3555, -278.9328,  153.6535, -241.8802, -315.1961, -210.0862,  -23.6923],
            [ 180.8046,  -57.3144, -110.9336,   37.0305,  158.0435,  -87.0602,  137.0495,  178.5903,  119.0350,   13.4241],
            [-79.8264,  25.3047,  48.9779, -16.3492, -69.7772,  38.4376, -60.5082, -78.8488, -52.5547,  -5.9268],
            [ 28.0060,  -8.8778, -17.1832,   5.7359,  24.4804, -13.4853,  21.2285,  27.6630,  18.4381,   2.0793],
            [-7.9605,  2.5235,  4.8842, -1.6304, -6.9584,  3.8331, -6.0340, -7.8630, -5.2409, -0.5910],
            [ 1.8689, -0.5924, -1.1467,  0.3828,  1.6336, -0.8999,  1.4166,  1.8460,  1.2304,  0.1388],
            [-0.3694,  0.1171,  0.2267, -0.0757, -0.3229,  0.1779, -0.2800, -0.3649, -0.2432, -0.0274],
            [ 0.0627, -0.0199, -0.0385,  0.0128,  0.0548, -0.0302,  0.0475,  0.0619,  0.0413,  0.0047],
            [-0.0093,  0.0030,  0.0057, -0.0019, -0.0082,  0.0045, -0.0071, -0.0092, -0.0061, -0.0007],
            [     0.0012,     -0.0004,     -0.0008,      0.0003,      0.0011,     -0.0006,      0.0009,      0.0012,
                  0.0008,      0.0001],
            [    -0.0002,      0.0000,      0.0001,     -0.0000,     -0.0001,      0.0001,     -0.0001,     -0.0002,
                  -0.0001,     -0.0000],
            [     0.0000,     -0.0000,     -0.0000,      0.0000,      0.0000,     -0.0000,      0.0000,      0.0000,
                  0.0000,      0.0000],
            [    -0.0000,      0.0000,      0.0000,     -0.0000,     -0.0000,      0.0000,     -0.0000,     -0.0000,
                  -0.0000,     -0.0000],
            [     0.0000,     -0.0000,     -0.0000,      0.0000,      0.0000,     -0.0000,      0.0000,      0.0000,
                  0.0000,      0.0000],
            [    -0.0000,      0.0000,      0.0000,     -0.0000,     -0.0000,      0.0000,     -0.0000,     -0.0000,
                  -0.0000,     -0.0000],
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
        let scheduler = EulerDiscreteScheduler(stepCount: 20, useKarrasSigmas: true)
        let generator: RandomGenerator = TorchRandomGenerator(seed: 50)
        let stdev = scheduler.initNoiseSigma
        var latent: MLShapedArray<Float32> = generator.nextArray(shape: [1, 4, 64, 64], mean: 0, stdev: stdev)
        zip(latent[0][0][0][0..<10].scalars, [
            -16.9352,   5.3684,  10.3907,  -3.4685, -14.8032,   8.1545, -12.8368, -16.7278, -11.1495,  -1.2574
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        let expectedOutputs: [[Float32]] = [
            [ 47.7743, -15.1443, -29.3121,   9.7846,  41.7601, -23.0040,  36.2128,  47.1892,  31.4528,   3.5471],
            [-114.0475,   36.1527,   69.9744,  -23.3580,  -99.6903,   54.9156,  -86.4477, -112.6508,  -75.0846,   -8.4676],
            [ 223.1332,  -70.7325, -136.9045,   45.6998,  195.0434, -107.4421,  169.1344,  220.4006,  146.9025,   16.5668],
            [-355.7216,  112.7626,  218.2548,  -72.8551, -310.9406,  171.2854, -269.6362, -351.3652, -234.1938,  -26.4110],
            [ 458.7301, -145.4159, -281.4562,   93.9522,  400.9816, -220.8856,  347.7164,  453.1123,  302.0107,   34.0590],
            [-474.8264,  150.5184,  291.3321,  -97.2489, -415.0515,  228.6362, -359.9173, -469.0114, -312.6079,  -35.2541],
            [ 391.2132, -124.0133, -240.0308,   80.1241,  341.9642, -188.3752,  296.5387,  386.4221,  257.5601,   29.0462],
            [-254.2596,   80.5994,  156.0022,  -52.0747, -222.2514,  122.4299, -192.7282, -251.1458, -167.3950,  -18.8778],
            [129.0889, -40.9207, -79.2031,  26.4386, 112.8382, -62.1583,  97.8491, 127.5080,  84.9873,   9.5844],
            [-50.6579,  16.0584,  31.0814, -10.3752, -44.2806,  24.3925, -38.3985, -50.0375, -33.3512,  -3.7612],
            [15.1893, -4.8150, -9.3195,  3.1109, 13.2772, -7.3139, 11.5135, 15.0033, 10.0001,  1.1278],
            [-3.4362,  1.0893,  2.1083, -0.7038, -3.0036,  1.6546, -2.6046, -3.3941, -2.2623, -0.2551],
            [ 0.5784, -0.1834, -0.3549,  0.1185,  0.5056, -0.2785,  0.4384,  0.5713,  0.3808,  0.0429],
            [-0.0714,  0.0226,  0.0438, -0.0146, -0.0624,  0.0344, -0.0541, -0.0705, -0.0470, -0.0053],
            [ 0.0063, -0.0020, -0.0039,  0.0013,  0.0055, -0.0031,  0.0048,  0.0063,  0.0042,  0.0005],
            [    -0.0004,      0.0001,      0.0002,     -0.0001,     -0.0003,      0.0002,     -0.0003,     -0.0004,
                  -0.0003,     -0.0000],
            [     0.0000,     -0.0000,     -0.0000,      0.0000,      0.0000,     -0.0000,      0.0000,      0.0000,
                  0.0000,      0.0000],
            [    -0.0000,      0.0000,      0.0000,     -0.0000,     -0.0000,      0.0000,     -0.0000,     -0.0000,
                  -0.0000,     -0.0000],
            [    -0.0000,      0.0000,      0.0000,     -0.0000,     -0.0000,      0.0000,     -0.0000,     -0.0000,
                  -0.0000,     -0.0000],
            [    -0.0000,      0.0000,      0.0000,     -0.0000,     -0.0000,      0.0000,     -0.0000,     -0.0000,
                  -0.0000,     -0.0000],
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
