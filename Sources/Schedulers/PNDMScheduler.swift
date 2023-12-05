//
//  PNDMScheduler.swift
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
///  [Hugging Face Diffusers PNDMScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py)
///
/// This scheduler uses the pseudo linear multi-step (PLMS) method only, skipping pseudo Runge-Kutta (PRK) steps
public final class PNDMScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let finalAlphaCumProd: Float
    public let timeSteps: [Double]
    public let predictionType: PredictionType
    
    public let alpha_t: [Float]
    public let sigma_t: [Float]
    public let lambda_t: [Float]

    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []

    // Internal state
    var counter: Int
    var ets: [MLShapedArray<Float32>]
    var currentSample: MLShapedArray<Float32>?

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
        setAlphaToOne: Bool? = nil,
        stepsOffset: Int? = nil,
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
        
        if setAlphaToOne ?? false {
            self.finalAlphaCumProd = 1
        } else {
            self.finalAlphaCumProd = alphasCumProd[0]
        }
        
        // Currently we only support VP-type noise shedule
        self.alpha_t = vForce.sqrt(self.alphasCumProd)
        self.sigma_t = vForce.sqrt(vDSP.subtract([Float](repeating: 1, count: self.alphasCumProd.count), self.alphasCumProd))
        self.lambda_t = zip(self.alpha_t, self.sigma_t).map { α, σ in log(α) - log(σ) }
        
        var timeSteps: [Double]
        switch timestepSpacing {
        case .linspace:
            timeSteps = linspace(0, Double(trainStepCount - 1), stepCount)
                .map { $0.rounded() }
                .reversed()
        case .leading:
            let stepRatio = trainStepCount / stepCount
            timeSteps = (0..<stepCount).map { Double($0 * stepRatio).rounded() + Double(stepsOffset ?? 0) }.reversed()
        case .trailing:
            let stepRatio = Double(trainStepCount) / Double(stepCount)
            timeSteps = stride(from: Double(trainStepCount), to: 1, by: -stepRatio).map { round($0) - 1 }
        }
        if timeSteps.count > 1 {
            // Repeat second timestep
            timeSteps.insert(timeSteps[1], at: 1)
        }
        
        if let strength {
            let initTimestep = min(Int(Float(stepCount) * strength), stepCount)
            let tStart = min(timeSteps.count - 1, max(stepCount - initTimestep, 0))
            timeSteps = Array(timeSteps[tStart..<timeSteps.count])
        }
        self.timeSteps = timeSteps
        self.counter = 0
        self.ets = []
        self.currentSample = nil
    }
    
    public func step(
        output: MLShapedArray<Float32>,
        timeStep t: Double,
        sample s: MLShapedArray<Float32>,
        generator: RandomGenerator
    ) -> MLShapedArray<Float32> {
        
        var timeStep = Int(t)
        let stepInc = (trainStepCount / inferenceStepCount)
        var prevStep = timeStep - stepInc
        var modelOutput = output
        var sample = s

        if counter != 1 {
            if ets.count > 3 {
                ets = Array(ets[(ets.count - 3)..<ets.count])
            }
            ets.append(output)
        } else {
            prevStep = timeStep
            timeStep = timeStep + stepInc
        }

        if ets.count == 1 && counter == 0 {
            modelOutput = output
            currentSample = sample
        } else if ets.count == 1 && counter == 1 {
            modelOutput = [output,  ets[back: 1]]
                .weightedSum([1.0/2.0, 1.0/2.0])
            sample = currentSample!
            currentSample = nil
        } else if ets.count == 2 {
            modelOutput = [ets[back: 1], ets[back: 2]]
                .weightedSum([3.0/2.0, -1.0/2.0])
        } else if ets.count == 3 {
            modelOutput = [ets[back: 1], ets[back: 2], ets[back: 3]]
                .weightedSum([23.0/12.0, -16.0/12.0, 5.0/12.0])
        } else {
            modelOutput = [ets[back: 1], ets[back: 2], ets[back: 3], ets[back: 4]]
                .weightedSum([55.0/24.0, -59.0/24.0, 37/24.0, -9/24.0])
        }
        
        let convertedOutput = convertModelOutput(modelOutput: modelOutput, timestep: timeStep, sample: sample)
        modelOutputs.removeAll(keepingCapacity: true)
        modelOutputs.append(convertedOutput)

        let prevSample = previousSample(sample, timeStep, prevStep, modelOutput)
        counter += 1
        return prevSample
    }
    
    /// Convert the model output to the corresponding type the algorithm needs.
    func convertModelOutput(modelOutput: MLShapedArray<Float32>, timestep: Int, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        assert(modelOutput.scalarCount == sample.scalarCount)
        let scalarCount = modelOutput.scalarCount
        let (alpha_t, sigma_t) = (self.alpha_t[timestep], self.sigma_t[timestep])
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

    /// Compute sample (denoised image) at previous step given a current time step
    ///
    /// - Parameters:
    ///   - sample: The current input to the model x_t
    ///   - timeStep: The current time step t
    ///   - prevStep: The previous time step t−δ
    ///   - modelOutput: Predicted noise residual the current time step e_θ(x_t, t)
    /// - Returns: Computes previous sample x_(t−δ)
    func previousSample(
        _ sample: MLShapedArray<Float32>,
        _ timeStep: Int,
        _ prevStep: Int,
        _ modelOutput: MLShapedArray<Float32>
    ) ->  MLShapedArray<Float32> {

        // Compute x_(t−δ) using formula (9) from
        // "Pseudo Numerical Methods for Diffusion Models on Manifolds",
        // Luping Liu, Yi Ren, Zhijie Lin & Zhou Zhao.
        // ICLR 2022
        //
        // Notation:
        //
        // alphaProdt       α_t
        // alphaProdtPrev   α_(t−δ)
        // betaProdt        (1 - α_t)
        // betaProdtPrev    (1 - α_(t−δ))
        let alphaProdt = alphasCumProd[timeStep]
        let alphaProdtPrev = prevStep >= 0 ? alphasCumProd[prevStep] : finalAlphaCumProd
        let betaProdt = 1 - alphaProdt
        let betaProdtPrev = 1 - alphaProdtPrev

        // sampleCoeff = (α_(t−δ) - α_t) divided by
        // denominator of x_t in formula (9) and plus 1
        // Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        // sqrt(α_(t−δ)) / sqrt(α_t))
        let sampleCoeff = sqrt(alphaProdtPrev / alphaProdt)
        
        var output = modelOutput
        switch predictionType {
        case .epsilon:
            // Do nothing
            break
        case .vPrediction:
            let scalarCount = modelOutput.scalarCount
            let alphaProdtPow = pow(alphaProdt, 0.5)
            let betaProdtPow = pow(betaProdt, 0.5)
            output = MLShapedArray(unsafeUninitializedShape: modelOutput.shape) { scalars, _ in
                assert(scalars.count == scalarCount)
                modelOutput.withUnsafeShapedBufferPointer { modelOutput, _, _ in
                    sample.withUnsafeShapedBufferPointer { sample, _, _ in
                        for i in 0..<scalarCount {
                            scalars.initializeElement(at: i, to: alphaProdtPow * modelOutput[i] + betaProdtPow * sample[i])
                        }
                    }
                }
            }
        }

        // Denominator of e_θ(x_t, t) in formula (9)
        let modelOutputDenomCoeff = alphaProdt * sqrt(betaProdtPrev)
        + sqrt(alphaProdt * betaProdt * alphaProdtPrev)

        // full formula (9)
        let modelCoeff = -(alphaProdtPrev - alphaProdt)/modelOutputDenomCoeff
        let prevSample = [sample, output]
            .weightedSum([Double(sampleCoeff), Double(modelCoeff)])

        return prevSample
    }
}
