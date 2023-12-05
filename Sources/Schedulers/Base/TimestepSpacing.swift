//
//  TimestepSpacing.swift
//
//
//  Created by Guillermo Cique Fernández on 30/11/23.
//

import Foundation

/// The way the timesteps should be scaled.
public enum TimestepSpacing: String, Decodable {
    case linspace
    case leading
    case trailing
    
    public init(from decoder: Decoder) throws {
        do {
            let container = try decoder.singleValueContainer()
            let rawValue = try container.decode(String.self)
            if let value = Self(rawValue: rawValue) {
                self = value
            } else {
                self = .leading
            }
        } catch {
            self = .leading
        }
    }
}
