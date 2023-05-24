// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Schedulers",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "Schedulers", targets: ["Schedulers"]),
    ],
    dependencies: [
        .package(url: "https://github.com/GuernikaCore/RandomGenerator.git", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "Schedulers",
            dependencies: [.product(name: "RandomGenerator", package: "RandomGenerator")]
        ),
        .testTarget(
            name: "SchedulersTests",
            dependencies: ["Schedulers"]
        ),
    ]
)
