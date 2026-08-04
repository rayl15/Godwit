// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Godwit",
    platforms: [
        .macOS(.v26),
    ],
    products: [
        .library(name: "Godwit", targets: ["Godwit"]),
        .executable(name: "godwit", targets: ["GodwitCLI"]),
    ],
    targets: [
        .target(
            name: "Godwit",
            path: "Sources/Godwit",
            resources: [.copy("Metal")]
        ),
        .executableTarget(
            name: "GodwitCLI",
            dependencies: ["Godwit"],
            path: "Sources/GodwitCLI"
        ),
        .testTarget(
            name: "GodwitTests",
            dependencies: ["Godwit"],
            path: "Tests/GodwitTests"
        ),
    ]
)
