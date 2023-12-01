//
//  KDPM2DiscreteScheduler.swift
//
//
//  Created by Guillermo Cique Fernández on 21/04/23.
//

import CoreML
import Accelerate
import RandomGenerator

/// A scheduler used to compute a de-noised image
///
///  This implementation matches:
///  [Hugging Face Diffusers DPMSolverMultistepScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_k_dpm_2_discrete.py)
///
/// Scheduler created by @crowsonkb in [k_diffusion](https://github.com/crowsonkb/k-diffusion), see:
/// https://github.com/crowsonkb/k-diffusion/blob/5b3af030dd83e0297272d861c19477735d0317ec/k_diffusion/sampling.py#L188
///
/// Scheduler inspired by DPM-Solver-2 and Algorthim 2 from Karras et al. (2022).
public final class KDPM2DiscreteScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let timeSteps: [Double]
    public let sigmas: [Double]
    public let sigmasInterpol: [Double]
    public let predictionType: PredictionType
    public let initNoiseSigma: Float
    public var order: Int { 2 }
    
    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []
    
    private(set) var sample: MLShapedArray<Float32>?
    private var stateInFirstOrder: Bool { sample == nil }
    
    /// Create a scheduler that uses a second order DPM-Solver++ algorithm.
    ///
    /// - Parameters:
    ///   - stepCount: Number of inference steps to schedule
    ///   - trainStepCount: Number of training diffusion steps
    ///   - betaSchedule: Method to schedule betas from betaStart to betaEnd
    ///   - betaStart: The starting value of beta for inference
    ///   - betaEnd: The end value for beta for inference
    /// - Returns: A scheduler ready for its first step
    public init(
        strength: Float? = nil,
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012,
        predictionType: PredictionType = .epsilon,
        timestepSpacing: TimestepSpacing? = nil,
        useKarrasSigmas: Bool = false
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        self.predictionType = predictionType
        let timestepSpacing = timestepSpacing ?? .linspace
        
        self.betas = betaSchedule.betas(betaStart: betaStart, betaEnd: betaEnd, trainStepCount: trainStepCount)
        self.alphas = betas.map({ 1.0 - $0 })
        var alphasCumProd = self.alphas
        for i in 1..<alphasCumProd.count {
            alphasCumProd[i] *= alphasCumProd[i -  1]
        }
        self.alphasCumProd = alphasCumProd
        
        var timeSteps: [Double]
        switch timestepSpacing {
        case .linspace:
            timeSteps = linspace(0, Double(trainStepCount - 1), stepCount)
                .reversed()
        case .leading:
            let stepRatio = trainStepCount / stepCount
            timeSteps = (0..<stepCount).map { Double($0 * stepRatio) }.reversed()
        case .trailing:
            let stepRatio = Double(trainStepCount) / Double(stepCount)
            timeSteps = stride(from: Double(trainStepCount), to: 1, by: -stepRatio).map { round($0) - 1 }
        }
        
        var sigmas: [Double] = alphasCumProd.map { Double(pow((1 - $0) / $0, 0.5)) }
        let logSigmas = sigmas.map { log($0) }
        sigmas = vDSP.linearInterpolate(elementsOf: sigmas, using: timeSteps)
        if useKarrasSigmas {
            sigmas = KDPM2DiscreteScheduler.convertToKarras(sigmas: sigmas, stepCount: stepCount)
            timeSteps = KDPM2DiscreteScheduler.convertToTimesteps(sigmas: sigmas, logSigmas: logSigmas)
        }
        sigmas = sigmas + [0]
        
        // interpolate sigmas
        var sigmasInterpol = vDSP.linearInterpolate(
            sigmas.map { log($0) },
            ([sigmas.last!] + sigmas[0..<sigmas.count - 1]).map { log($0) },
            using: 0.5
        ).map { exp($0) }.map { $0.isNaN ? 0 : $0 }
        
        let timestepsInterpol = KDPM2DiscreteScheduler.convertToTimesteps(sigmas: sigmasInterpol, logSigmas: logSigmas)
        print(timestepsInterpol)
        
        switch timestepSpacing {
        case .linspace, .leading:
            self.initNoiseSigma = Float(sigmas.max() ?? 1)
        case .trailing:
            self.initNoiseSigma = pow(pow(Float(sigmas.max() ?? 1), 2) + 1, 0.5)
        }
        
        sigmas = [sigmas[0]] +
            sigmas[1..<sigmas.count].flatMap { [$0, $0] } +
            [sigmas.last!]
        sigmasInterpol = [sigmasInterpol[0]] +
            sigmasInterpol[1..<sigmasInterpol.count].flatMap { [$0, $0] } +
            [sigmasInterpol.last!]
        
        timeSteps = [timeSteps[0]] + zip(timestepsInterpol[1..<timestepsInterpol.count - 1], timeSteps[1..<timeSteps.count]).flatMap { [$0, $1] }
        if let strength {
            let initTimestep = min(Int(Float(stepCount) * strength), stepCount)
            let tStart = min(timeSteps.count - 1, max(stepCount - initTimestep, 0) * 2) // * order
            timeSteps = Array(timeSteps[tStart..<timeSteps.count])
            sigmas = Array(sigmas[tStart..<sigmas.count])
            sigmasInterpol = Array(sigmasInterpol[tStart..<sigmasInterpol.count])
        }
        self.timeSteps = timeSteps.map { Double($0) }
        self.sigmas = sigmas
        self.sigmasInterpol = sigmasInterpol
    }
    
    func indexForTimestep(_ timeStep: Double) -> Int {
        return timeSteps.firstIndex(of: timeStep) ?? timeSteps.count - 1
    }

    public func scaleModelInput(timeStep: Double, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        let stepIndex = indexForTimestep(timeStep)
        let sigma: Float32 = Float32(stateInFirstOrder ? sigmas[stepIndex] : sigmasInterpol[stepIndex])
        let scale: Float32 = pow(pow(sigma, 2) + 1, 0.5)
        return MLShapedArray(unsafeUninitializedShape: sample.shape) { scalars, _ in
            sample.withUnsafeShapedBufferPointer { sample, _, _ in
                vDSP.divide(sample, scale, result: &scalars)
            }
        }
    }

    public func step(
        output: MLShapedArray<Float32>,
        timeStep t: Double,
        sample s: MLShapedArray<Float32>,
        generator: RandomGenerator
    ) -> MLShapedArray<Float32> {
        let stepIndex = indexForTimestep(t)
        var sample: MLShapedArray<Float32> = s
        let scalarCount = sample.scalarCount
        
        let sigma: Float32
        let sigmaInterpol: Float32
        let sigmaNext: Float32
        if stateInFirstOrder {
            sigma = Float32(sigmas[stepIndex])
            sigmaInterpol = Float32(sigmasInterpol[stepIndex + 1])
            sigmaNext = Float32(sigmas[stepIndex + 1])
        } else {
            // 2nd order / KDPM2's method
            sigma = Float32(sigmas[stepIndex - 1])
            sigmaInterpol = Float32(sigmasInterpol[stepIndex])
            sigmaNext = Float32(sigmas[stepIndex])
        }
        let sigmaInput = stateInFirstOrder ? sigma : sigmaInterpol
        
        let predOriginalSample: MLShapedArray<Float32>
        switch predictionType {
        case .epsilon:
            predOriginalSample = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
                sample.withUnsafeShapedBufferPointer { sample, _, _ in
                    output.withUnsafeShapedBufferPointer { output, _, _ in
                        for i in 0..<scalarCount {
                            scalars.initializeElement(at: i, to: sample[i] - sigmaInput * output[i])
                        }
                    }
                }
            }
        case .vPrediction:
            // * c_out + input * c_skip
            let sigmaPow: Float32 = pow(sigmaInput, 2) + 1
            let sigmaAux: Float32 = -sigmaInput / pow(sigmaPow, 0.5)
            predOriginalSample = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
                sample.withUnsafeShapedBufferPointer { sample, _, _ in
                    output.withUnsafeShapedBufferPointer { output, _, _ in
                        for i in 0..<scalarCount {
                            scalars.initializeElement(at: i, to: output[i] * sigmaAux + (sample[i] / sigmaPow))
                        }
                    }
                }
            }
        }
        
        modelOutputs.removeAll(keepingCapacity: true)
        modelOutputs.append(predOriginalSample)
        
        let prevSample: MLShapedArray<Float32>
        if stateInFirstOrder {
            // 3. delta timestep
            let dt: Float32 = sigmaInterpol - sigma
            
            // store for 2nd order step
            self.sample = sample
            
            prevSample = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
                sample.withUnsafeShapedBufferPointer { sample, _, _ in
                    predOriginalSample.withUnsafeShapedBufferPointer { original, _, _ in
                        for i in 0..<scalarCount {
                            // 2. Convert to an ODE derivative for 1st order
                            let derivative = (sample[i] - original[i]) / sigma
                            scalars.initializeElement(at: i, to: sample[i] + derivative * dt)
                        }
                    }
                }
            }
        } else {
            // DPM-Solver-2
            // 3. delta timestep
            let dt: Float32 = sigmaNext - sigma
            
            sample = self.sample!
            self.sample = nil
            
            prevSample = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
                sample.withUnsafeShapedBufferPointer { sample, _, _ in
                    predOriginalSample.withUnsafeShapedBufferPointer { original, _, _ in
                        for i in 0..<scalarCount {
                            // 2. Convert to an ODE derivative for 2nd order
                            let derivative = (sample[i] - original[i]) / sigmaInterpol
                            scalars.initializeElement(at: i, to: sample[i] + derivative * dt)
                        }
                    }
                }
            }
        }
        
        return prevSample
    }
    
    public func addNoise(
        originalSample: MLShapedArray<Float32>,
        noise: [MLShapedArray<Float32>],
        timeStep t: Double?
    ) -> [MLShapedArray<Float32>] {
        let stepIndex = t.flatMap { timeSteps.firstIndex(of: $0) } ?? 0
        let sigma = Float32(sigmas[stepIndex])
        let noisySamples = noise.map { noise in
            MLShapedArray(unsafeUninitializedShape: originalSample.shape) { scalars, _ in
                originalSample.withUnsafeShapedBufferPointer { sample, _, _ in
                    noise.withUnsafeShapedBufferPointer { noise, _, _ in
                        for i in 0..<originalSample.scalarCount {
                            scalars.initializeElement(at: i, to: sample[i] + noise[i] * sigma)
                        }
                    }
                }
            }
        }
        return noisySamples
    }
}
