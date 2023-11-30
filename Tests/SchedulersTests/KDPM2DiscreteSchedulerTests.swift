//
//  KDPM2DiscreteSchedulerTests.swift
//  
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import XCTest
@testable import Schedulers

final class KDPM2DiscreteSchedulerTests: XCTestCase {
    func test1StepsLinspace() throws {
        let scheduler = KDPM2DiscreteScheduler(stepCount: 1, timestepSpacing: .linspace)
        XCTAssertEqual(scheduler.timeSteps, [0])
    }
    
    func test2StepsLinspace() throws {
        let scheduler = KDPM2DiscreteScheduler(stepCount: 2, timestepSpacing: .linspace)
        zip(scheduler.timeSteps, [999.00000000, 233.20971680,   0]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test33StepsLinspace() throws {
        let scheduler = KDPM2DiscreteScheduler(stepCount: 33, timestepSpacing: .linspace)
        print(scheduler.sigmas)
        print(scheduler.sigmasInterpol)
        zip(scheduler.timeSteps, [
            999.0000, 983.5647, 967.7812, 952.3476, 936.5625, 921.1296, 905.3438, 889.9105, 874.1250, 858.6903, 842.9062,
            827.4695, 811.6875, 796.2476, 780.4688, 765.0242, 749.2500, 733.7988, 718.0312, 702.5723, 686.8125, 671.3444,
            655.5938, 640.1148, 624.3750, 608.8832, 593.1562, 577.6496, 561.9375, 546.4146, 530.7188, 515.1776, 499.5000,
            483.9385, 468.2812, 452.6969, 437.0625, 421.4530, 405.8438, 390.2062, 374.6250, 358.9561, 343.4062, 327.7014,
            312.1875, 296.4409, 280.9688, 265.1727, 249.7500, 233.8935, 218.5312, 202.5977, 187.3125, 171.2754, 156.0938,
            139.9077, 124.8750, 108.4546,  93.6562,  76.8053,  62.4375,  44.5434,  31.2188,   4.8832,   0.000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test33StepsLeading() throws {
        let scheduler = KDPM2DiscreteScheduler(stepCount: 33, timestepSpacing: .leading)
        zip(scheduler.timeSteps, [
            960.0000, 945.1619, 930.0000, 915.1622, 900.0000, 885.1620, 870.0000, 855.1609, 840.0000, 825.1589, 810.0000,
            795.1558, 780.0000, 765.1516, 750.0000, 735.1461, 720.0000, 705.1393, 690.0000, 675.1312, 660.0000, 645.1217,
            630.0000, 615.1107, 600.0000, 585.0980, 570.0000, 555.0840, 540.0000, 525.0682, 510.0000, 495.0507, 480.0000,
            465.0315, 450.0000, 435.0101, 420.0000, 404.9864, 390.0000, 374.9601, 360.0000, 344.9304, 330.0000, 314.8965,
            300.0000, 284.8571, 270.0000, 254.8103, 240.0000, 224.7528, 210.0000, 194.6792, 180.0000, 164.5800, 150.0000,
            134.4374, 120.0000, 104.2114,  90.0000,  73.7981,  60.0000,  42.7957,  30.0000,   4.7689,   0.0000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test33StepsTrailing() throws {
        let scheduler = KDPM2DiscreteScheduler(stepCount: 33, timestepSpacing: .trailing)
        zip(scheduler.timeSteps, [
            999.0000, 984.1605, 969.0000, 953.6725, 938.0000, 923.1622, 908.0000, 893.1621, 878.0000, 862.6721, 847.0000,
            832.1594, 817.0000, 802.1566, 787.0000, 772.1526, 757.0000, 741.6573, 726.0000, 711.1408, 696.0000, 681.1329,
            666.0000, 650.6318, 635.0000, 620.1125, 605.0000, 590.1003, 575.0000, 559.5920, 544.0000, 529.0704, 514.0000,
            499.0532, 484.0000, 469.0341, 454.0000, 438.5135, 423.0000, 407.9889, 393.0000, 377.9628, 363.0000, 347.4285,
            332.0000, 316.8989, 302.0000, 286.8600, 272.0000, 256.3002, 241.0000, 225.7549, 211.0000, 195.6820, 181.0000,
            165.5839, 151.0000, 134.9017, 120.0000, 104.2114,  90.0000,  73.7981,  60.0000,  42.1086,  29.0000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test33StepsKarras() throws {
        let scheduler = KDPM2DiscreteScheduler(stepCount: 33, useKarrasSigmas: true)
        print(scheduler.timeSteps)
        print(scheduler.sigmas)
        print(scheduler.sigmasInterpol)
    }
    
    func test50StepsLinspace() throws {
        var scheduler = KDPM2DiscreteScheduler(stepCount: 50, timestepSpacing: .linspace)
        zip(scheduler.timeSteps, [
            999.0000, 988.8805, 978.6122, 968.4934, 958.2245, 948.1058, 937.8367, 927.7183, 917.4490, 907.3304, 897.0612,
            886.9426, 876.6735, 866.5547, 856.2857, 846.1663, 835.8979, 825.7779, 815.5102, 805.3893, 795.1224, 785.0004,
            774.7347, 764.6113, 754.3469, 744.2217, 733.9592, 723.8320, 713.5714, 703.4421, 693.1837, 683.0518, 672.7959,
            662.6613, 652.4081, 642.2703, 632.0204, 621.8791, 611.6326, 601.4877, 591.2449, 581.0958, 570.8571, 560.7037,
            550.4694, 540.3111, 530.0816, 519.9182, 509.6939, 499.5251, 489.3061, 479.1313, 468.9184, 458.7373, 448.5306,
            438.3429, 428.1429, 417.9479, 407.7551, 397.5524, 387.3673, 377.1562, 366.9796, 356.7594, 346.5918, 336.3619,
            326.2041, 315.9632, 305.8163, 295.5637, 285.4286, 275.1624, 265.0408, 254.7595, 244.6531, 234.3545, 224.2653,
            213.9462, 203.8775, 193.5343, 183.4898, 173.1166, 163.1020, 152.6915, 142.7143, 132.2547, 122.3265, 111.7999,
            101.9388,  91.3149,  81.5510,  70.7725,  61.1633,  50.1022,  40.7755,  29.0375,  20.3878,   3.7423,   0.0000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        scheduler = KDPM2DiscreteScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .linspace)
        zip(scheduler.timeSteps, [
            693.1837, 683.0518, 672.7959, 662.6613, 652.4081, 642.2703, 632.0204, 621.8791, 611.6326, 601.4877, 591.2449,
            581.0958, 570.8571, 560.7037, 550.4694, 540.3111, 530.0816, 519.9182, 509.6939, 499.5251, 489.3061, 479.1313,
            468.9184, 458.7373, 448.5306, 438.3429, 428.1429, 417.9479, 407.7551, 397.5524, 387.3673, 377.1562, 366.9796,
            356.7594, 346.5918, 336.3619, 326.2041, 315.9632, 305.8163, 295.5637, 285.4286, 275.1624, 265.0408, 254.7595,
            244.6531, 234.3545, 224.2653, 213.9462, 203.8775, 193.5343, 183.4898, 173.1166, 163.1020, 152.6915, 142.7143,
            132.2547, 122.3265, 111.7999, 101.9388,  91.3149,  81.5510,  70.7725,  61.1633,  50.1022,  40.7755,  29.0375,
            20.3878,   3.7423,   0.0000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test50StepsLeading() throws {
        var scheduler = KDPM2DiscreteScheduler(stepCount: 50, timestepSpacing: .leading)
        zip(scheduler.timeSteps, [
            980.0000, 970.0717, 960.0000, 950.0718, 940.0000, 930.0720, 920.0000, 910.0722, 900.0000, 890.0720, 880.0000,
            870.0718, 860.0000, 850.0714, 840.0000, 830.0708, 820.0000, 810.0700, 800.0000, 790.0690, 780.0000, 770.0677,
            760.0000, 750.0662, 740.0000, 730.0645, 720.0000, 710.0624, 700.0000, 690.0602, 680.0000, 670.0576, 660.0000,
            650.0549, 640.0000, 630.0517, 620.0000, 610.0483, 600.0000, 590.0446, 580.0000, 570.0405, 560.0000, 550.0362,
            540.0000, 530.0316, 520.0000, 510.0266, 500.0000, 490.0212, 480.0000, 470.0154, 460.0000, 450.0094, 440.0000,
            430.0028, 420.0000, 409.9958, 400.0000, 389.9883, 380.0000, 369.9802, 360.0000, 349.9714, 340.0000, 329.9619,
            320.0000, 309.9514, 300.0000, 289.9397, 280.0000, 269.9267, 260.0000, 249.9119, 240.0000, 229.8950, 220.0000,
            209.8751, 200.0000, 189.8513, 180.0000, 169.8222, 160.0000, 149.7855, 140.0000, 129.7375, 120.0000, 109.6718,
            100.0000,  89.5766,  80.0000,  69.4252,  60.0000,  49.1481,  40.0000,  28.4888,  20.0000,   3.6994,   0.0000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        scheduler = KDPM2DiscreteScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .leading)
        zip(scheduler.timeSteps, [
            680.0000, 670.0576, 660.0000, 650.0549, 640.0000, 630.0517, 620.0000, 610.0483, 600.0000, 590.0446, 580.0000,
            570.0405, 560.0000, 550.0362, 540.0000, 530.0316, 520.0000, 510.0266, 500.0000, 490.0212, 480.0000, 470.0154,
            460.0000, 450.0094, 440.0000, 430.0028, 420.0000, 409.9958, 400.0000, 389.9883, 380.0000, 369.9802, 360.0000,
            349.9714, 340.0000, 329.9619, 320.0000, 309.9514, 300.0000, 289.9397, 280.0000, 269.9267, 260.0000, 249.9119,
            240.0000, 229.8950, 220.0000, 209.8751, 200.0000, 189.8513, 180.0000, 169.8222, 160.0000, 149.7855, 140.0000,
            129.7375, 120.0000, 109.6718, 100.0000,  89.5766,  80.0000,  69.4252,  60.0000,  49.1481,  40.0000,  28.4888,
            20.0000,   3.6994,   0.0000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test50StepsTrailing() throws {
        var scheduler = KDPM2DiscreteScheduler(stepCount: 50, timestepSpacing: .trailing)
        zip(scheduler.timeSteps, [
            999.0000, 989.0712, 979.0000, 969.0715, 959.0000, 949.0720, 939.0000, 929.0721, 919.0000, 909.0721, 899.0000,
            889.0720, 879.0000, 869.0718, 859.0000, 849.0714, 839.0000, 829.0707, 819.0000, 809.0699, 799.0000, 789.0689,
            779.0000, 769.0676, 759.0000, 749.0662, 739.0000, 729.0643, 719.0000, 709.0624, 699.0000, 689.0601, 679.0000,
            669.0575, 659.0000, 649.0547, 639.0000, 629.0516, 619.0000, 609.0482, 599.0000, 589.0444, 579.0000, 569.0404,
            559.0000, 549.0360, 539.0000, 529.0312, 519.0000, 509.0263, 499.0000, 489.0209, 479.0000, 469.0152, 459.0000,
            449.0091, 439.0000, 429.0025, 419.0000, 408.9954, 399.0000, 388.9879, 379.0000, 368.9798, 359.0000, 348.9710,
            339.0000, 328.9614, 319.0000, 308.9508, 299.0000, 288.9391, 279.0000, 268.9260, 259.0000, 248.9111, 239.0000,
            228.8940, 219.0000, 208.8740, 199.0000, 188.8500, 179.0000, 168.8205, 159.0000, 148.7834, 139.0000, 128.7347,
            119.0000, 108.6679,  99.0000,  88.5706,  79.0000,  68.4152,  59.0000,  48.1281,  39.0000,  27.4306,  19.0000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
        scheduler = KDPM2DiscreteScheduler(strength: 0.7, stepCount: 50, timestepSpacing: .trailing)
        zip(scheduler.timeSteps, [
            699.0000, 689.0601, 679.0000, 669.0575, 659.0000, 649.0547, 639.0000, 629.0516, 619.0000, 609.0482, 599.0000,
            589.0444, 579.0000, 569.0404, 559.0000, 549.0360, 539.0000, 529.0312, 519.0000, 509.0263, 499.0000, 489.0209,
            479.0000, 469.0152, 459.0000, 449.0091, 439.0000, 429.0025, 419.0000, 408.9954, 399.0000, 388.9879, 379.0000,
            368.9798, 359.0000, 348.9710, 339.0000, 328.9614, 319.0000, 308.9508, 299.0000, 288.9391, 279.0000, 268.9260,
            259.0000, 248.9111, 239.0000, 228.8940, 219.0000, 208.8740, 199.0000, 188.8500, 179.0000, 168.8205, 159.0000,
            148.7834, 139.0000, 128.7347, 119.0000, 108.6679,  99.0000,  88.5706,  79.0000,  68.4152,  59.0000,  48.1281,
            39.0000,  27.4306,  19.0000
        ]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }
    
    func test50StepsKarras() throws {
        let scheduler = KDPM2DiscreteScheduler(stepCount: 50, useKarrasSigmas: true)
        print(scheduler.timeSteps)
        print(scheduler.sigmas)
        print(scheduler.sigmasInterpol)
    }
}