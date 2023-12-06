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
    case eulerDiscreteKarras
    case eulerAncenstralDiscrete
    case dpmSolverMultistep
    case dpmSolverMultistepKarras
    case dpmSolverSinglestep
    case dpmSolverSinglestepKarras
    case dpm2
    case dpm2Karras
    case lcm
    
    public var id: String { rawValue }
    
    public var description: String {
        switch self {
        case .ddim:
            return "DDIM"
        case .pndm:
            return "PNDM"
        case .eulerDiscrete:
            return "Euler"
        case .eulerDiscreteKarras:
            return "Euler Karras"
        case .eulerAncenstralDiscrete:
            return "EulerA"
        case .dpmSolverMultistep:
            return "DPM++ 2M"
        case .dpmSolverMultistepKarras:
            return "DPM++ 2M Karras"
        case .dpmSolverSinglestep:
            return "DPM++ SDE"
        case .dpmSolverSinglestepKarras:
            return "DPM++ SDE Karras"
        case .dpm2:
            return "DPM2"
        case .dpm2Karras:
            return "DPM2 Karras"
        case .lcm:
            return "LCM"
        }
    }
    
    public func create(
        strength: Float? = nil,
        stepCount: Int = 50,
        originalStepCount: Int? = nil,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012,
        setAlphaToOne: Bool? = nil,
        stepsOffset: Int? = nil,
        predictionType: PredictionType = .epsilon,
        timestepSpacing: TimestepSpacing? = nil
    ) -> any Scheduler {
        switch self {
        case .ddim: return DDIMScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            setAlphaToOne: setAlphaToOne,
            stepsOffset: stepsOffset,
            predictionType: predictionType,
            timestepSpacing: timestepSpacing
        )
        case .pndm: return PNDMScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            setAlphaToOne: setAlphaToOne,
            stepsOffset: stepsOffset,
            predictionType: predictionType,
            timestepSpacing: timestepSpacing
        )
        case .eulerDiscrete: return EulerDiscreteScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            stepsOffset: stepsOffset,
            predictionType: predictionType,
            timestepSpacing: timestepSpacing
        )
        case .eulerDiscreteKarras: return EulerDiscreteScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            stepsOffset: stepsOffset,
            predictionType: predictionType,
            timestepSpacing: timestepSpacing,
            useKarrasSigmas: true
        )
        case .eulerAncenstralDiscrete: return EulerAncestralDiscreteScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            stepsOffset: stepsOffset,
            predictionType: predictionType,
            timestepSpacing: timestepSpacing
        )
        case .dpmSolverMultistep: return DPMSolverMultistepScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            stepsOffset: stepsOffset,
            predictionType: predictionType,
            timestepSpacing: timestepSpacing
        )
        case .dpmSolverMultistepKarras: return DPMSolverMultistepScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            stepsOffset: stepsOffset,
            predictionType: predictionType,
            timestepSpacing: timestepSpacing,
            useKarrasSigmas: true
        )
        case .dpmSolverSinglestep: return DPMSolverSinglestepScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            predictionType: predictionType
        )
        case .dpmSolverSinglestepKarras: return DPMSolverSinglestepScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            predictionType: predictionType,
            useKarrasSigmas: true
        )
        case .dpm2: return KDPM2DiscreteScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            stepsOffset: stepsOffset,
            predictionType: predictionType,
            timestepSpacing: timestepSpacing
        )
        case .dpm2Karras: return KDPM2DiscreteScheduler(
            strength: strength,
            stepCount: stepCount,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            stepsOffset: stepsOffset,
            predictionType: predictionType,
            timestepSpacing: timestepSpacing,
            useKarrasSigmas: true
        )
        case .lcm: return LCMScheduler(
            strength: strength,
            stepCount: stepCount,
            originalStepCount: originalStepCount ?? 50,
            trainStepCount: trainStepCount,
            betaSchedule: betaSchedule,
            betaStart: betaStart,
            betaEnd: betaEnd,
            setAlphaToOne: setAlphaToOne,
            predictionType: predictionType
        )
        }
    }
}
