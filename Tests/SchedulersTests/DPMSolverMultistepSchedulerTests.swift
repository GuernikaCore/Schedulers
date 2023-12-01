//
//  DPMSolverMultistepSchedulerTests.swift
//
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import XCTest
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
}
