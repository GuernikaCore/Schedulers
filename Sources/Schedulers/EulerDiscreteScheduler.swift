//
//  EulerDiscreteScheduler.swift
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
///  [Hugging Face Diffusers EulerDiscreteScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py)
///
/// Euler scheduler (Algorithm 2) from Karras et al. (2022) https://arxiv.org/abs/2206.00364. . Based on the original
/// k-diffusion implementation by Katherine Crowson:
/// https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L51
public final class EulerDiscreteScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let timeSteps: [Double]
    public let sigmas: [Double]
    public let predictionType: PredictionType
    public let initNoiseSigma: Float

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
        predictionType: PredictionType = .epsilon
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        self.predictionType = predictionType

        self.betas = betaSchedule.betas(betaStart: betaStart, betaEnd: betaEnd, trainStepCount: trainStepCount)
        self.alphas = betas.map({ 1.0 - $0 })
        var alphasCumProd = self.alphas
        for i in 1..<alphasCumProd.count {
            alphasCumProd[i] *= alphasCumProd[i -  1]
        }
        self.alphasCumProd = alphasCumProd
        
        var timeSteps = linspace(0, Double(trainStepCount - 1), inferenceStepCount)
        var sigmas: [Double] = alphasCumProd.map { Double(pow((1 - $0) / $0, 0.5)) }
        sigmas = vDSP.linearInterpolate(elementsOf: sigmas, using: timeSteps).reversed() + [0]
        self.initNoiseSigma = Float(sigmas.max() ?? 1)
        if let strength {
            let tEnc = Int(Float(timeSteps.count) * strength)
            let startSigma = timeSteps.count - tEnc
            timeSteps = Array(timeSteps[0..<tEnc])
            sigmas = Array(sigmas[startSigma..<sigmas.count])
        }
        timeSteps.reverse()
        self.timeSteps = timeSteps
        self.sigmas = sigmas
    }
    
    public func scaleModelInput(timeStep t: Double, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
        let sigma = Float32(sigmas[stepIndex])
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
        sample: MLShapedArray<Float32>,
        generator: RandomGenerator
    ) -> MLShapedArray<Float32> {
        let scalarCount = sample.scalarCount
        
        let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
        let sigma = Float32(sigmas[stepIndex])
        
        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        let predOriginalSample: MLShapedArray<Float32>
        switch predictionType {
        case .epsilon:
            predOriginalSample = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
                sample.withUnsafeShapedBufferPointer { sample, _, _ in
                    output.withUnsafeShapedBufferPointer { output, _, _ in
                        for i in 0..<scalarCount {
                            scalars.initializeElement(at: i, to: sample[i] - sigma * output[i])
                        }
                    }
                }
            }
        case .vPrediction:
            // * c_out + input * c_skip
            let sigmaPow: Float32 = pow(sigma, 2) + 1
            let sigmaAux: Float32 = -sigma / pow(sigmaPow, 0.5)
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
        
        // 2. Convert to an ODE derivative
        let dt: Float32 = Float32(sigmas[stepIndex + 1]) - sigma
        return MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
            sample.withUnsafeShapedBufferPointer { sample, _, _ in
                predOriginalSample.withUnsafeShapedBufferPointer { original, _, _ in
                    for i in 0..<scalarCount {
                        let derivative = (sample[i] - original[i]) / sigma
                        scalars.initializeElement(at: i, to: sample[i] + derivative * dt)
                    }
                }
            }
        }
    }
    
    public func addNoise(
        originalSample: MLShapedArray<Float32>,
        noise: [MLShapedArray<Float32>],
        timeStep t: Double
    ) -> [MLShapedArray<Float32>] {
        let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
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
