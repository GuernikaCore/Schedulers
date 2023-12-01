//
//  DDIMScheduler.swift
//  
//
//  Created by Guillermo Cique Fernández on 21/04/23.
//

import CoreML
import RandomGenerator

/// A scheduler used to compute a de-noised image
///
///  This implementation matches:
///  [Hugging Face Diffusers DDIMScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py)
///
/// Denoising diffusion implicit models is a scheduler that extends the denoising procedure introduced in denoising
/// diffusion probabilistic models (DDPMs) with non-Markovian guidance.
public final class DDIMScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let timeSteps: [Double]
    public let predictionType: PredictionType
    
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
        predictionType: PredictionType = .epsilon,
        timestepSpacing: TimestepSpacing? = nil
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        self.predictionType = predictionType
        let timestepSpacing = timestepSpacing ?? .leading

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
                .map { $0.rounded() }
                .reversed()
        case .leading:
            let stepRatio = trainStepCount / stepCount
            timeSteps = (0..<stepCount).map { Double($0 * stepRatio).rounded() }.reversed()
        case .trailing:
            let stepRatio = Double(trainStepCount) / Double(stepCount)
            timeSteps = stride(from: Double(trainStepCount), to: 1, by: -stepRatio).map { round($0) - 1 }
        }
        
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
        let scalarCount = sample.scalarCount
        
        // See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        // Ideally, read DDIM paper in-detail understanding

        // Notation (<variable name> -> <name in paper>
        // - pred_noise_t -> e_theta(x_t, t)
        // - pred_original_sample -> f_theta(x_t, t) or x_0
        // - std_dev_t -> sigma_t
        // - eta -> η
        // - pred_sample_direction -> "direction pointing to x_t"
        // - pred_prev_sample -> "x_t-1"

        // 1. get previous step value (=t-1)
        let timeStep = Int(t)
        let stepInc = (trainStepCount / inferenceStepCount)
        let prevStep = timeStep - stepInc
        
        // 2. compute alphas, betas
        let alphaProdt = alphasCumProd[timeStep]
        let alphaProdtPrev = alphasCumProd[max(0, prevStep)]
        
        let betaProdt = 1 - alphaProdt
        
        // 3. compute predicted original sample from predicted noise also called
        // "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        let predOriginalSample: MLShapedArray<Float32>
        let predEpsilon: MLShapedArray<Float32>
        switch predictionType {
        case .epsilon:
            let alphaProdtPow = pow(alphaProdt, 0.5)
            let betaProdtPow = pow(betaProdt, 0.5)
            predOriginalSample = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
                sample.withUnsafeShapedBufferPointer { sample, _, _ in
                    output.withUnsafeShapedBufferPointer { output, _, _ in
                        for i in 0..<scalarCount {
                            scalars.initializeElement(at: i, to: (sample[i] - betaProdtPow * output[i]) / alphaProdtPow)
                        }
                    }
                }
            }
            predEpsilon = output
        case .vPrediction:
            let alphaProdtPow = pow(alphaProdt, 0.5)
            let betaProdtPow = pow(betaProdt, 0.5)
            predOriginalSample = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
                sample.withUnsafeShapedBufferPointer { sample, _, _ in
                    output.withUnsafeShapedBufferPointer { output, _, _ in
                        for i in 0..<scalarCount {
                            scalars.initializeElement(at: i, to: alphaProdtPow * sample[i] - betaProdtPow * output[i])
                        }
                    }
                }
            }
            predEpsilon = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
                sample.withUnsafeShapedBufferPointer { sample, _, _ in
                    output.withUnsafeShapedBufferPointer { output, _, _ in
                        for i in 0..<scalarCount {
                            scalars.initializeElement(at: i, to: alphaProdtPow * output[i] + betaProdtPow * sample[i])
                        }
                    }
                }
            }
        }
        
        modelOutputs.removeAll(keepingCapacity: true)
        modelOutputs.append(predOriginalSample)
        
        // 4. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        let predSampleDirectionAux = pow(1 - alphaProdtPrev, 0.5)
        
        // 5. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        let prevSampleAux = pow(alphaProdtPrev, 0.5)
        
        return MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
            predOriginalSample.withUnsafeShapedBufferPointer { original, _, _ in
                predEpsilon.withUnsafeShapedBufferPointer { epsilon, _, _ in
                    for i in 0..<scalarCount {
                        let sampleDirection = predSampleDirectionAux * epsilon[i]
                        scalars.initializeElement(at: i, to: prevSampleAux * original[i] + sampleDirection)
                    }
                }
            }
        }
    }
}
