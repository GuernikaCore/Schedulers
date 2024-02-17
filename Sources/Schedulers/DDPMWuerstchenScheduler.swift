//
//  DDPMWuerstchenScheduler.swift
//
//
//  Created by Guillermo Cique Fernández on 2/1/23.
//

import CoreML
import Accelerate
import RandomGenerator

/// A scheduler used to compute a de-noised image
///
///  This implementation matches:
///  [Hugging Face Diffusers DDPMWuerstchenScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm_wuerstchen.py)
///
/// Denoising diffusion probabilistic models (DDPMs) explores the connections between denoising score matching and
/// Langevin dynamics sampling.
public final class DDPMWuerstchenScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let timeSteps: [Double]
    
    let initAlphaCumprod: Double
    let scaler: Double
    let s: Double
    
    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []
    
    /// Create a scheduler that uses a pseudo linear multi-step (PLMS)  method
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
        scaler: Double = 1,
        s: Double = 0.008
    ) {
        self.trainStepCount = 0
        self.inferenceStepCount = stepCount
        
        self.betas = betaSchedule.betas(betaStart: betaStart, betaEnd: betaEnd, trainStepCount: trainStepCount)
        self.alphas = betas.map({ 1.0 - $0 })
        var alphasCumProd = self.alphas
        for i in 1..<alphasCumProd.count {
            alphasCumProd[i] *= alphasCumProd[i -  1]
        }
        self.alphasCumProd = alphasCumProd
        
        self.initAlphaCumprod = pow(cos(s / (1 + s) * .pi * 0.5), 2)
        self.scaler = scaler
        self.s = s
        
        var timeSteps: [Double] = linspace(1, 0, inferenceStepCount + 1)
            .dropLast()
        
        if let strength {
            let initTimestep = min(Int(Float(stepCount) * strength), stepCount)
            let tStart = min(timeSteps.count - 1, max(stepCount - initTimestep, 0))
            timeSteps = Array(timeSteps[tStart..<timeSteps.count])
        }
        self.timeSteps = timeSteps
    }
    
    public func step(
        output: MLShapedArray<Float32>,
        timeStep t: Double,
        sample: MLShapedArray<Float32>,
        generator: RandomGenerator
    ) -> MLShapedArray<Float32> {
        let timeStep = t
        let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
        let prevTimestep = stepIndex == timeSteps.count - 1 ? 0 : timeSteps[stepIndex + 1]
        
        let alphaCumprod = _alphaCumprod(timeStep: timeStep)
        let alphaCumprodPrev = _alphaCumprod(timeStep: prevTimestep)
        let alpha = Float(alphaCumprod / alphaCumprodPrev)
        
        let sqrtAlphaCumprod = Float(sqrt(1 - alphaCumprod))
        let sqrtAlpha = Float(sqrt(1 / alpha))
        
        let scalarCount = sample.scalarCount
        var mu = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
            output.withUnsafeShapedBufferPointer { output, _, _ in
                sample.withUnsafeShapedBufferPointer { sample, _, _ in
                    for i in 0..<scalarCount {
                        scalars.initializeElement(
                            at: i,
                            to: sqrtAlpha * (sample[i] - (1 - alpha) * output[i] / sqrtAlphaCumprod)
                        )
                    }
                }
            }
        }
        var stdNoise: MLShapedArray<Float32> = generator.nextArray(shape: mu.shape, mean: 0, stdev: 1)
        if prevTimestep != 0 {
            let stdNoiseAux = sqrt((1-alpha) * Float(1-alphaCumprodPrev) / Float(1-alphaCumprod))
            stdNoise.withUnsafeMutableShapedBufferPointer { pointer, _, _ in
                vDSP.multiply(stdNoiseAux, pointer, result: &pointer)
            }
            mu = MLShapedArray(unsafeUninitializedShape: mu.shape) { scalars, _ in
                mu.withUnsafeShapedBufferPointer { mu, _, _ in
                    stdNoise.withUnsafeShapedBufferPointer { noise, _, _ in
                        for i in 0..<scalarCount {
                            scalars.initializeElement(at: i, to: mu[i] + noise[i])
                        }
                    }
                }
            }
        }
        
        return mu
    }
    
    func _alphaCumprod(timeStep t: Double) -> Double {
        var timeStep = t
        if scaler > 1 {
            timeStep = pow(1 - (1 - t), scaler)
        } else if scaler < 1 {
            timeStep = pow(t, scaler)
        }
        let alphaCumprod = pow(cos(
            (timeStep + s) / (1 + s) * .pi * 0.5
        ), 2) / initAlphaCumprod
        return min(max(alphaCumprod, 0.0001), 0.9999)
    }
}
