//
//  DDIMSchedulerTests.swift
//
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import XCTest
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
}
