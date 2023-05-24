//
//  BetaSchedule.swift
//  
//
//  Created by Guillermo Cique Fernández on 20/5/23.
//

import Foundation

/// How to map a beta range to a sequence of betas to step over
public enum BetaSchedule {
    /// Linear stepping between start and end
    case linear
    /// Steps using linspace(sqrt(start),sqrt(end))^2
    case scaledLinear
    
    func betas(betaStart: Float, betaEnd: Float, trainStepCount: Int) -> [Float] {
        switch self {
        case .linear:
            return linspace(betaStart, betaEnd, trainStepCount)
        case .scaledLinear:
            return linspace(pow(betaStart, 0.5), pow(betaEnd, 0.5), trainStepCount).map({ $0 * $0 })
        }
    }
}
