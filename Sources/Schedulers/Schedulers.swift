//
//  Schedulers.swift
//
//
//  Created by Guillermo Cique Fernández on 20/5/23.
//

public enum Schedulers: String, CaseIterable, Identifiable, CustomStringConvertible {
    case ddim
    case pndm
    case eulerDiscrete
    case eulerAncenstralDiscrete
    case dpmSolverMultistep
    case dpm2Karras
    
    public var id: String { rawValue }
    
    public var description: String {
        switch self {
        case .ddim:
            return "DDIM"
        case .pndm:
            return "PNDM"
        case .eulerDiscrete:
            return "Euler"
        case .eulerAncenstralDiscrete:
            return "EulerA"
        case .dpmSolverMultistep:
            return "DPM-Solver++"
        case .dpm2Karras:
            return "DPM2 Karras"
        }
    }
    
    public func create(
        strength: Float? = nil,
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012,
        predictionType: PredictionType = .epsilon
    ) -> any Scheduler {
        switch self {
        case .ddim: return DDIMScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            predictionType: predictionType
        )
        case .pndm: return PNDMScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            predictionType: predictionType
        )
        case .eulerDiscrete: return EulerDiscreteScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            predictionType: predictionType
        )
        case .eulerAncenstralDiscrete: return EulerAncestralDiscreteScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            predictionType: predictionType
        )
        case .dpmSolverMultistep: return DPMSolverMultistepScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            predictionType: predictionType
        )
        case .dpm2Karras: return KDPM2DiscreteScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            predictionType: predictionType
        )
        }
    }
}
