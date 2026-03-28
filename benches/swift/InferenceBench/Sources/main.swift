import CoreML
import Foundation

// MARK: - Argument Parsing

struct BenchConfig {
    let modelPath: String
    let iterations: Int
    let warmup: Int
    let jsonOutput: Bool
}

func parseArgs() -> BenchConfig {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        fputs("""
        Usage: InferenceBench <model_path.mlmodelc> [--iterations N] [--warmup W] [--json]

        Options:
          --iterations N   Number of timed iterations (default: 100)
          --warmup W       Number of warmup iterations (default: 10)
          --json           Output results as JSON
        """, stderr)
        exit(1)
    }

    let modelPath = args[1]
    var iterations = 100
    var warmup = 10
    var jsonOutput = false

    var i = 2
    while i < args.count {
        switch args[i] {
        case "--iterations":
            i += 1
            guard i < args.count, let n = Int(args[i]) else {
                fputs("Error: --iterations requires an integer argument\n", stderr)
                exit(1)
            }
            iterations = n
        case "--warmup":
            i += 1
            guard i < args.count, let w = Int(args[i]) else {
                fputs("Error: --warmup requires an integer argument\n", stderr)
                exit(1)
            }
            warmup = w
        case "--json":
            jsonOutput = true
        default:
            fputs("Warning: unknown argument '\(args[i])'\n", stderr)
        }
        i += 1
    }

    return BenchConfig(
        modelPath: modelPath,
        iterations: iterations,
        warmup: warmup,
        jsonOutput: jsonOutput
    )
}

// MARK: - Dummy Input Generation

func createDummyInput(for model: MLModel) throws -> MLDictionaryFeatureProvider {
    let inputDesc = model.modelDescription.inputDescriptionsByName
    var features: [String: MLFeatureValue] = [:]

    for (name, desc) in inputDesc {
        guard let constraint = desc.multiArrayConstraint else {
            fputs("Warning: input '\(name)' is not a MultiArray, skipping\n", stderr)
            continue
        }

        let shape = constraint.shape
        let multiArray = try MLMultiArray(shape: shape, dataType: .float32)
        let count = multiArray.count
        let ptr = multiArray.dataPointer.bindMemory(to: Float32.self, capacity: count)
        for j in 0..<count {
            ptr[j] = Float32.random(in: -1.0...1.0)
        }
        features[name] = MLFeatureValue(multiArray: multiArray)
    }

    return try MLDictionaryFeatureProvider(dictionary: features)
}

// MARK: - Statistics

struct BenchResults {
    let modelPath: String
    let loadTimeMs: Double
    let firstInferenceMs: Double
    let meanMs: Double
    let medianMs: Double
    let p95Ms: Double
    let p99Ms: Double
    let minMs: Double
    let maxMs: Double
    let totalTimeMs: Double
    let iterations: Int
    let warmup: Int
}

func computeResults(
    modelPath: String,
    loadTime: Double,
    firstInference: Double,
    latencies: [Double],
    warmup: Int
) -> BenchResults {
    let sorted = latencies.sorted()
    let count = sorted.count
    let mean = sorted.reduce(0, +) / Double(count)
    let p50 = sorted[count / 2]
    let p95 = sorted[min(Int(Double(count) * 0.95), count - 1)]
    let p99 = sorted[min(Int(Double(count) * 0.99), count - 1)]
    let total = sorted.reduce(0, +)

    return BenchResults(
        modelPath: modelPath,
        loadTimeMs: loadTime * 1000,
        firstInferenceMs: firstInference * 1000,
        meanMs: mean * 1000,
        medianMs: p50 * 1000,
        p95Ms: p95 * 1000,
        p99Ms: p99 * 1000,
        minMs: (sorted.first ?? 0) * 1000,
        maxMs: (sorted.last ?? 0) * 1000,
        totalTimeMs: total * 1000,
        iterations: count,
        warmup: warmup
    )
}

// MARK: - Output Formatting

func printJSON(_ results: BenchResults) {
    let json: [String: Any] = [
        "model_path": results.modelPath,
        "iterations": results.iterations,
        "warmup": results.warmup,
        "load_time_ms": round(results.loadTimeMs * 1000) / 1000,
        "first_inference_ms": round(results.firstInferenceMs * 1000) / 1000,
        "mean_ms": round(results.meanMs * 1000) / 1000,
        "median_ms": round(results.medianMs * 1000) / 1000,
        "p95_ms": round(results.p95Ms * 1000) / 1000,
        "p99_ms": round(results.p99Ms * 1000) / 1000,
        "min_ms": round(results.minMs * 1000) / 1000,
        "max_ms": round(results.maxMs * 1000) / 1000,
        "total_time_ms": round(results.totalTimeMs * 1000) / 1000,
    ]
    if let data = try? JSONSerialization.data(
        withJSONObject: json, options: [.prettyPrinted, .sortedKeys]),
        let str = String(data: data, encoding: .utf8)
    {
        print(str)
    }
}

func printTable(_ results: BenchResults) {
    let name = (results.modelPath as NSString).lastPathComponent

    func fmt(_ ms: Double) -> String {
        if ms < 1.0 {
            return String(format: "%.1fµs", ms * 1000)
        } else if ms < 1000 {
            return String(format: "%.2fms", ms)
        } else {
            return String(format: "%.2fs", ms / 1000)
        }
    }

    print("")
    print("Inference Benchmark: \(name)")
    print(String(repeating: "─", count: 44))
    print("  Model load time:".padding(toLength: 26, withPad: " ", startingAt: 0) + fmt(results.loadTimeMs))
    print("  First inference:".padding(toLength: 26, withPad: " ", startingAt: 0) + fmt(results.firstInferenceMs))
    print("  Iterations:".padding(toLength: 26, withPad: " ", startingAt: 0) + "\(results.iterations) (warmup: \(results.warmup))")
    print(String(repeating: "─", count: 44))
    print("  Mean:".padding(toLength: 26, withPad: " ", startingAt: 0) + fmt(results.meanMs))
    print("  Median (p50):".padding(toLength: 26, withPad: " ", startingAt: 0) + fmt(results.medianMs))
    print("  p95:".padding(toLength: 26, withPad: " ", startingAt: 0) + fmt(results.p95Ms))
    print("  p99:".padding(toLength: 26, withPad: " ", startingAt: 0) + fmt(results.p99Ms))
    print("  Min:".padding(toLength: 26, withPad: " ", startingAt: 0) + fmt(results.minMs))
    print("  Max:".padding(toLength: 26, withPad: " ", startingAt: 0) + fmt(results.maxMs))
    print(String(repeating: "─", count: 44))
    print("  Total time:".padding(toLength: 26, withPad: " ", startingAt: 0) + fmt(results.totalTimeMs))
    print("")
}

// MARK: - Main

let config = parseArgs()

let modelURL = URL(fileURLWithPath: config.modelPath)
guard FileManager.default.fileExists(atPath: config.modelPath) else {
    fputs("Error: model not found at '\(config.modelPath)'\n", stderr)
    exit(1)
}

// Load model
let startLoad = CFAbsoluteTimeGetCurrent()
let model: MLModel
do {
    model = try MLModel(contentsOf: modelURL)
} catch {
    fputs("Error loading model: \(error)\n", stderr)
    exit(1)
}
let loadTime = CFAbsoluteTimeGetCurrent() - startLoad

// Create dummy input
let input: MLDictionaryFeatureProvider
do {
    input = try createDummyInput(for: model)
} catch {
    fputs("Error creating dummy input: \(error)\n", stderr)
    exit(1)
}

// Warmup
for _ in 0..<config.warmup {
    let _ = try? model.prediction(from: input)
}

// Timed runs
var latencies: [Double] = []
latencies.reserveCapacity(config.iterations)

var firstInference: Double = 0

for i in 0..<config.iterations {
    let start = CFAbsoluteTimeGetCurrent()
    let _ = try? model.prediction(from: input)
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    latencies.append(elapsed)
    if i == 0 {
        firstInference = elapsed
    }
}

let results = computeResults(
    modelPath: config.modelPath,
    loadTime: loadTime,
    firstInference: firstInference,
    latencies: latencies,
    warmup: config.warmup
)

if config.jsonOutput {
    printJSON(results)
} else {
    printTable(results)
}
