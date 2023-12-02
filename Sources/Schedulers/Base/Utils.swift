//
//  Utils.swift
//  
//
//  Created by Guillermo Cique Fernández on 21/5/23.
//

import CoreML
import Accelerate
import Foundation

/// Evenly spaced floats between specified interval
///
/// - Parameters:
///   - start: Start of the interval
///   - end: End of the interval
///   - count: The number of floats to return between [*start*, *end*]
///   - endpoint: If true, `end` is the last sample. Otherwise, it is not included. Default is true
/// - Returns: Float array with *count* elements evenly spaced between at *start* and *end*
func linspace<Number: FloatingPoint>(_ start: Number, _ end: Number, _ count: Int, endpoint: Bool = true) -> [Number] {
    guard count > 1 else { return [start] }
    let div = endpoint ? Number(count - 1) : Number(count)
    let scale = (end - start) / div
    var y = (0..<count).map { Number($0) * scale + start }
    if endpoint {
        y[count - 1] = end
    }
    return y
}

extension Collection {
    /// Collection element index from the back. *self[back: 1]* yields the last element
    public subscript(back i: Int) -> Element {
        return self[index(endIndex, offsetBy: -i)]
    }
}

extension Array where Element == MLShapedArray<Float32> {
    /// Compute weighted sum of shaped arrays of equal shapes
    ///
    /// - Parameters:
    ///   - weights: The weights each array is multiplied by
    /// - Returns: sum_i weights[i]*values[i]
    func weightedSum(_ weights: [Double]) -> MLShapedArray<Float32> {
        let scalarCount = self.first!.scalarCount
        assert(weights.count > 1 && self.count == weights.count)
        assert(self.allSatisfy({ $0.scalarCount == scalarCount }))

        return MLShapedArray(unsafeUninitializedShape: self.first!.shape) { scalars, _ in
            scalars.initialize(repeating: 0.0)
            for i in 0..<self.count {
                let w = Float(weights[i])
                self[i].withUnsafeShapedBufferPointer { buffer, _, _ in
                    assert(buffer.count == scalarCount)
                    // scalars[j] = w * values[i].scalars[j]
                    cblas_saxpy(Int32(scalarCount), w, buffer.baseAddress, 1, scalars.baseAddress, 1)
                }
            }
        }
    }
}

extension FloatingPoint {
    func clipped(to range: ClosedRange<Self>) -> Self {
        return min(max(self, range.lowerBound), range.upperBound)
    }
}
