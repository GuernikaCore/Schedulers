//
//  DPMSolverMultistepScheduler.swift
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
///  [Hugging Face Diffusers DPMSolverMultistepScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py)
///
/// It uses the DPM-Solver++ algorithm: [code](https://github.com/LuChengTHU/dpm-solver) [paper](https://arxiv.org/abs/2211.01095).
/// Limitations:
///  - Only implemented for DPM-Solver++ algorithm (not DPM-Solver).
///  - Second order only.
///  - No dynamic thresholding.
///  - `midpoint` solver algorithm.
public final class DPMSolverMultistepScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Double]
    public let alphas: [Double]
    public let alphasCumProd: [Double]
    public let timeSteps: [Double]
    public let predictionType: PredictionType

    public let alpha_t: [Double]
    public let sigma_t: [Double]
    public let lambda_t: [Double]
    
    public let solverOrder = 2
    private(set) var lowerOrderStepped = 0
    
    /// Whether to use lower-order solvers in the final steps. Only valid for less than 15 inference steps.
    /// We empirically find this trick can stabilize the sampling of DPM-Solver, especially with 10 or fewer steps.
    public let useLowerOrderFinal = true
    
    // Stores solverOrder (2) items
    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []

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
        betaStart: Double = 0.00085,
        betaEnd: Double = 0.012,
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

        // Currently we only support VP-type noise shedule
        self.alpha_t = vForce.sqrt(self.alphasCumProd)
        self.sigma_t = vForce.sqrt(vDSP.subtract([Double](repeating: 1, count: self.alphasCumProd.count), self.alphasCumProd))
        self.lambda_t = zip(self.alpha_t, self.sigma_t).map { α, σ in log(α) - log(σ) }

        var timeSteps: [Int] = linspace(0, Float(trainStepCount - 1), stepCount+1).dropFirst().map { Int(round($0)) }
        if let strength {
            let tEnc = Int(Float(timeSteps.count) * strength)
            timeSteps = Array(timeSteps[0..<tEnc])
        }
        timeSteps.reverse()
        self.timeSteps = timeSteps.map { Double($0) }
    }
    
    /// Convert the model output to the corresponding type the algorithm needs.
    /// This implementation is for second-order DPM-Solver++.
    func convertModelOutput(modelOutput: MLShapedArray<Float32>, timestep: Int, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        assert(modelOutput.scalarCount == sample.scalarCount)
        let scalarCount = modelOutput.scalarCount
        let (alpha_t, sigma_t) = (Float32(self.alpha_t[timestep]), Float32(self.sigma_t[timestep]))
        // This could be optimized with a Metal kernel if we find we need to
        switch predictionType {
        case .epsilon:
            return MLShapedArray(unsafeUninitializedShape: modelOutput.shape) { scalars, _ in
                assert(scalars.count == scalarCount)
                modelOutput.withUnsafeShapedBufferPointer { modelOutput, _, _ in
                    sample.withUnsafeShapedBufferPointer { sample, _, _ in
                        for i in 0..<scalarCount {
                            scalars.initializeElement(at: i, to: (sample[i] - modelOutput[i] * sigma_t) / alpha_t)
                        }
                    }
                }
            }
        case .vPrediction:
            return MLShapedArray(unsafeUninitializedShape: modelOutput.shape) { scalars, _ in
                assert(scalars.count == scalarCount)
                modelOutput.withUnsafeShapedBufferPointer { modelOutput, _, _ in
                    sample.withUnsafeShapedBufferPointer { sample, _, _ in
                        for i in 0..<scalarCount {
                            scalars.initializeElement(at: i, to: alpha_t * sample[i] - sigma_t * modelOutput[i])
                        }
                    }
                }
            }
        }
    }

    /// One step for the first-order DPM-Solver (equivalent to DDIM).
    /// See https://arxiv.org/abs/2206.00927 for the detailed derivation.
    /// var names and code structure mostly follow https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
    func firstOrderUpdate(
        modelOutput: MLShapedArray<Float32>,
        timestep: Int,
        prevTimestep: Int,
        sample: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        let (p_lambda_t, lambda_s) = (Double(lambda_t[prevTimestep]), Double(lambda_t[timestep]))
        let p_alpha_t = Double(alpha_t[prevTimestep])
        let (p_sigma_t, sigma_s) = (Double(sigma_t[prevTimestep]), Double(sigma_t[timestep]))
        let h = p_lambda_t - lambda_s
        // x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        let x_t = [sample, modelOutput].weightedSum(
            [p_sigma_t / sigma_s, -p_alpha_t * expm1(-h)]
        )
        return x_t
    }

    /// One step for the second-order multistep DPM-Solver++ algorithm, using the midpoint method.
    /// var names and code structure mostly follow https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
    func secondOrderUpdate(
        modelOutputs: [MLShapedArray<Float32>],
        timesteps: [Double],
        prevTimestep t: Int,
        sample: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        let (s0, s1) = (Int(timesteps[back: 1]), Int(timesteps[back: 2]))
        let (m0, m1) = (modelOutputs[back: 1], modelOutputs[back: 2])
        let (p_lambda_t, lambda_s0, lambda_s1) = (Double(lambda_t[t]), Double(lambda_t[s0]), Double(lambda_t[s1]))
        let p_alpha_t = Double(alpha_t[t])
        let (p_sigma_t, sigma_s0) = (Double(sigma_t[t]), Double(sigma_t[s0]))
        let (h, h_0) = (p_lambda_t - lambda_s0, lambda_s0 - lambda_s1)
        let r0 = h_0 / h
        let D0 = m0
        
        // D1 = (1.0 / r0) * (m0 - m1)
        let D1 = [m0, m1].weightedSum([1/r0, -1/r0])
        
        // See https://arxiv.org/abs/2211.01095 for detailed derivations
        // x_t = (
        //     (sigma_t / sigma_s0) * sample
        //     - (alpha_t * (torch.exp(-h) - 1.0)) * D0
        //     - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
        // )
        let x_t = [sample, D0, D1].weightedSum(
            [p_sigma_t/sigma_s0, -p_alpha_t * expm1(-h), -0.5 * p_alpha_t * expm1(-h)]
        )
        return x_t
    }

    public func step(
        output: MLShapedArray<Float32>,
        timeStep t: Double,
        sample: MLShapedArray<Float32>,
        generator: RandomGenerator
    ) -> MLShapedArray<Float32> {
        let timeStep = Int(t)
        let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
        let prevTimestep = Int(stepIndex == timeSteps.count - 1 ? 0 : timeSteps[stepIndex + 1])

        let lowerOrderFinal = useLowerOrderFinal && stepIndex == timeSteps.count - 1 && timeSteps.count < 15
        let lowerOrderSecond = useLowerOrderFinal && stepIndex == timeSteps.count - 2 && timeSteps.count < 15
        let lowerOrder = lowerOrderStepped < 1 || lowerOrderFinal || lowerOrderSecond
        
        let modelOutput = convertModelOutput(modelOutput: output, timestep: timeStep, sample: sample)
        if modelOutputs.count == solverOrder { modelOutputs.removeFirst() }
        modelOutputs.append(modelOutput)
        
        let prevSample: MLShapedArray<Float32>
        if lowerOrder {
            prevSample = firstOrderUpdate(modelOutput: modelOutput, timestep: timeStep, prevTimestep: prevTimestep, sample: sample)
        } else {
            prevSample = secondOrderUpdate(
                modelOutputs: modelOutputs,
                timesteps: [timeSteps[stepIndex - 1], t],
                prevTimestep: prevTimestep,
                sample: sample
            )
        }
        if lowerOrderStepped < solverOrder {
            lowerOrderStepped += 1
        }
        
        return prevSample
    }
}
