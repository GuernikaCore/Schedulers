//
//  Scheduler.swift
//
//
//  Created by Guillermo Cique Fernández on 20/5/23.
//

import CoreML
import Accelerate
import RandomGenerator

public protocol Scheduler {
    /// Number of diffusion steps performed during training
    var trainStepCount: Int { get }

    /// Number of inference steps to be performed
    var inferenceStepCount: Int { get }

    /// Training diffusion time steps index by inference time step
    var timeSteps: [Double] { get }

    /// Schedule of betas which controls the amount of noise added at each timestep
    var betas: [Float] { get }

    /// 1 - betas
    var alphas: [Float] { get }

    /// Cached cumulative product of alphas
    var alphasCumProd: [Float] { get }
    
    var order: Int { get }

    /// Standard deviation of the initial noise distribution
    var initNoiseSigma: Float { get }
    
    /// Denoised latents
    var modelOutputs: [MLShapedArray<Float32>] { get }
    
    func scaleModelInput(timeStep: Double, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32>

    /// Compute a de-noised image sample and step scheduler state
    ///
    /// - Parameters:
    ///   - output: The predicted residual noise output of learned diffusion model
    ///   - timeStep: The current time step in the diffusion chain
    ///   - sample: The current input sample to the diffusion model
    /// - Returns: Predicted de-noised sample at the previous time step
    /// - Postcondition: The scheduler state is updated.
    ///   The state holds the current sample and history of model output noise residuals
    func step(
        output: MLShapedArray<Float32>,
        timeStep t: Double,
        sample s: MLShapedArray<Float32>,
        generator: RandomGenerator
    ) -> MLShapedArray<Float32>
    
    func addNoise(
        originalSample: MLShapedArray<Float32>,
        noise: [MLShapedArray<Float32>],
        timeStep t: Double?
    ) -> [MLShapedArray<Float32>]
}

public extension Scheduler {
    var order: Int { 1 }
    
    var initNoiseSigma: Float { 1 }
    
    /// Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
    /// current timestep.
    func scaleModelInput(timeStep: Double, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> { sample }
    
    func addNoise(
        originalSample: MLShapedArray<Float32>,
        noise: [MLShapedArray<Float32>]
    ) -> [MLShapedArray<Float32>] {
        return addNoise(originalSample: originalSample, noise: noise, timeStep: nil)
    }
    
    func addNoise(
        originalSample: MLShapedArray<Float32>,
        noise: [MLShapedArray<Float32>],
        timeStep t: Double?
    ) -> [MLShapedArray<Float32>] {
        let timeStep: Int = t.map { Int($0) } ?? Int(timeSteps[0])
        let alphaProdt = alphasCumProd[timeStep]
        let betaProdt = 1 - alphaProdt
        let sqrtAlphaProdt = sqrt(alphaProdt)
        let sqrtBetaProdt = sqrt(betaProdt)
        
        let noisySamples = noise.map {
            [originalSample, $0].weightedSum([Double(sqrtAlphaProdt), Double(sqrtBetaProdt)])
        }

        return noisySamples
    }
}

public extension Scheduler {
    static func convertToKarras(sigmas: [Double], stepCount: Int) -> [Double] {
        let sigmaMin: Double = sigmas.last!
        let sigmaMax = sigmas.first!
        let rho: Double = 7 // 7.0 is the value used in the paper
        let ramp: [Double] = linspace(0, 1, stepCount)
        let minInvRho: Double = pow(sigmaMin, (1 / rho))
        let maxInvRho: Double = pow(sigmaMax, (1 / rho))
        
        return ramp.map { pow(maxInvRho + $0 * (minInvRho - maxInvRho), rho) }
    }
    
    static func convertToTimesteps(sigmas: [Double], logSigmas: [Double]) -> [Double] {
        return sigmas.map { sigma in
            let logSigma = log(sigma)
            let dists = logSigmas.map { logSigma - $0 }
            
            // last index that is not negative, clipped to last index - 1
            var lowIndex = dists.reduce(-1) { partialResult, dist in
                return dist >= 0 && partialResult < dists.endIndex-2 ? partialResult + 1 : partialResult
            }
            lowIndex = max(lowIndex, 0)
            let highIndex = lowIndex + 1
            
            let low = logSigmas.count > lowIndex ? logSigmas[lowIndex] : logSigmas[0]
            let high = logSigmas.count > highIndex ? logSigmas[highIndex] : logSigmas[logSigmas.count - 1]
            
            // Interpolate sigmas
            let w = ((low - logSigma) / (low - high)).clipped(to: 0...1)
            
            // transform interpolated value to time range
            let t = (1 - w) * Double(lowIndex) + w * Double(highIndex)
            return t
        }
    }
}
