// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "InferenceBench",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "InferenceBench", path: "Sources")
    ]
)
