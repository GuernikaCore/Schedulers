//
//  EulerDiscreteSchedulerTests.swift
//  
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import XCTest
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
}
