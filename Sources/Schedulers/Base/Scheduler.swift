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
