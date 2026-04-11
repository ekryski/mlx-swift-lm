// swift-tools-version: 5.12
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import Foundation

// mlx-swift dependency.
//
// Default: pull ekryski/mlx-swift @ ek/speed-improvements-2 from GitHub. SPM
// will not initialize the nested git submodules (mlx, mlx-c) under
// Source/Cmlx, so the remote checkout cannot actually build — it is here as a
// placeholder so the manifest is portable.
//
// Local development: set MLX_SWIFT_PATH to a local clone of mlx-swift that
// has its submodules initialized. Example:
//
//   git clone --recursive -b ek/speed-improvements-2 \
//     https://github.com/ekryski/mlx-swift /path/to/mlx-swift
//   export MLX_SWIFT_PATH=/path/to/mlx-swift
//   swift build -c release
//
// Tests: scripts/build-metallib.sh probes the same env var so the metallib
// build picks up the same source tree.
let mlxSwiftDependency: Package.Dependency = {
    if let path = ProcessInfo.processInfo.environment["MLX_SWIFT_PATH"],
       !path.isEmpty {
        return .package(path: path)
    }
    return .package(
        url: "https://github.com/ekryski/mlx-swift",
        branch: "ek/speed-improvements-2"
    )
}()

let package = Package(
    name: "mlx-swift-lm",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "MLXLLM",
            targets: ["MLXLLM"]),
        .library(
            name: "MLXVLM",
            targets: ["MLXVLM"]),
        .library(
            name: "MLXLMCommon",
            targets: ["MLXLMCommon"]),
        .library(
            name: "MLXEmbedders",
            targets: ["MLXEmbedders"]),
    ],
    dependencies: [
        mlxSwiftDependency,
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMinor(from: "1.2.0")
        ),
    ],
    targets: [
        .target(
            name: "MLXLLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Libraries/MLXLLM",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXVLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Libraries/MLXVLM",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Libraries/MLXLMCommon",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .target(name: "MLXLMCommon"),
            ],
            path: "Libraries/MLXEmbedders",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "MLXLMTests",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
            ],
            path: "Tests/MLXLMTests",
            exclude: [
                "README.md"
            ],
            resources: [.process("Resources/1080p_30.mov"), .process("Resources/audio_only.mov")],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "MLXLMIntegrationTests",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
            ],
            path: "Tests/MLXLMIntegrationTests",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "Benchmarks",
            dependencies: [
                "MLXLLM",
                "MLXVLM",
                "MLXLMCommon",
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Tests/Benchmarks",
            resources: [.process("Resources/llm-test-prompts"), .process("Resources/wikitext2")],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "PrefillBench",
            dependencies: [
                "MLXLLM",
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/PrefillBench",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    // docc builder
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
