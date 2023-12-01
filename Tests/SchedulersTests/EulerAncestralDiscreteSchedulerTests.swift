//
//  EulerDiscreteSchedulerTests.swift
//  
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import XCTest
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
}
