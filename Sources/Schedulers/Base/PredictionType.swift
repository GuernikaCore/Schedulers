//
//  PredictionType.swift
//  
//
//  Created by Guillermo Cique Fernández on 20/5/23.
//

import Foundation

/// Prediction type of the scheduler function
public enum PredictionType: String, Decodable {
    /// Predicting the noise of the diffusion process
    case epsilon
    /// See section 2.4 https://imagen.research.google/video/paper.pdf
    case vPrediction = "v_prediction"
    
    public init(from decoder: Decoder) throws {
        do {
            let container = try decoder.singleValueContainer()
            let rawValue = try container.decode(String.self)
            if let value = Self(rawValue: rawValue) {
                self = value
            } else {
                self = .epsilon
            }
        } catch {
            self = .epsilon
        }
    }
}
