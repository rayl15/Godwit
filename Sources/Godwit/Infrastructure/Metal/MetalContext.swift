import Foundation
import Metal

public enum MetalError: Error, CustomStringConvertible {
    case noDevice
    case shaderNotFound(String)
    case compileFailed(shader: String, underlying: Error)
    case functionNotFound(shader: String, function: String)
    case bufferAllocationFailed(bytes: Int)
    case encoderCreationFailed

    public var description: String {
        switch self {
        case .noDevice:
            return "no Metal device"
        case .shaderNotFound(let name):
            return "shader '\(name).metal' not found in bundle"
        case .compileFailed(let shader, let underlying):
            return "compiling \(shader).metal failed: \(underlying)"
        case .functionNotFound(let shader, let function):
            return "\(shader).metal has no function '\(function)'"
        case .bufferAllocationFailed(let bytes):
            return "could not allocate \(bytes) bytes"
        case .encoderCreationFailed:
            return "could not create a command encoder"
        }
    }
}

/// Owns the Metal device, queue, and compiled pipelines.
///
/// Shaders are compiled from source at run time rather than built into a
/// metallib. That costs a little at startup and buys two things: editing a
/// kernel needs no build-system involvement, and the package builds on machines
/// with only the Command Line Tools, where `xcrun metal` does not exist.
///
/// Each `.metal` file becomes its own `MTLLibrary`, so files are free to define
/// helpers with the same names without colliding.
public final class MetalContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    /// Optional; when set, the runtime records where its time goes.
    public var profiler: Profiler?

    private var libraries: [String: MTLLibrary] = [:]
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
        guard let queue = device.makeCommandQueue() else { throw MetalError.noDevice }
        self.device = device
        self.queue = queue
    }

    /// Compiles `<name>.metal` from the resource bundle, caching the result.
    public func library(named name: String) throws -> MTLLibrary {
        lock.lock()
        defer { lock.unlock() }
        if let cached = libraries[name] { return cached }

        guard let url = Self.shaderURL(named: name) else {
            throw MetalError.shaderNotFound(name)
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        let options = MTLCompileOptions()
        options.mathMode = .fast

        do {
            let library = try device.makeLibrary(source: source, options: options)
            libraries[name] = library
            return library
        } catch {
            throw MetalError.compileFailed(shader: name, underlying: error)
        }
    }

    /// Returns a compute pipeline for `function` in `<shader>.metal`, cached.
    ///
    /// `constants` are Metal function constants — specialisation values baked in
    /// at pipeline build time. They are how a kernel stays model-agnostic
    /// without paying for it: a head dimension supplied this way is a compile
    /// time constant to the shader compiler, so loops still unroll, where the
    /// same value passed in a buffer would not.
    public func pipeline(
        shader: String, function: String, constants: [Int: Int] = [:],
        booleanConstants: [Int: Bool] = [:]
    ) throws -> MTLComputePipelineState {
        let key = constants.isEmpty && booleanConstants.isEmpty
            ? "\(shader).\(function)"
            : "\(shader).\(function)["
                + (constants.sorted { $0.key < $1.key }
                    .map { "\($0.key)=\($0.value)" }
                 + booleanConstants.sorted { $0.key < $1.key }
                    .map { "\($0.key)=\($0.value)" }).joined(separator: ",") + "]"
        lock.lock()
        if let cached = pipelines[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let library = try library(named: shader)
        let kernel: MTLFunction
        if constants.isEmpty && booleanConstants.isEmpty {
            guard let plain = library.makeFunction(name: function) else {
                throw MetalError.functionNotFound(shader: shader, function: function)
            }
            kernel = plain
        } else {
            let values = MTLFunctionConstantValues()
            for (index, value) in constants {
                var scalar = UInt32(value)
                values.setConstantValue(&scalar, type: .uint, index: index)
            }
            for (index, value) in booleanConstants {
                var flag = value
                values.setConstantValue(&flag, type: .bool, index: index)
            }
            kernel = try library.makeFunction(name: function, constantValues: values)
        }
        let state = try device.makeComputePipelineState(function: kernel)

        lock.lock()
        pipelines[key] = state
        lock.unlock()
        return state
    }

    /// Locates a `.metal` file anywhere under the bundle's copied `Metal` tree.
    private static func shaderURL(named name: String) -> URL? {
        let bundle = Bundle.module
        if let direct = bundle.url(forResource: name, withExtension: "metal") {
            return direct
        }
        guard let root = bundle.url(forResource: "Metal", withExtension: nil),
              let walker = FileManager.default.enumerator(at: root,
                                                          includingPropertiesForKeys: nil)
        else { return nil }
        for case let url as URL in walker
        where url.pathExtension == "metal" && url.deletingPathExtension().lastPathComponent == name {
            return url
        }
        return nil
    }

    /// Commits, waits, and records both wall and GPU time under `label`.
    public func run(_ label: String, _ commands: MTLCommandBuffer) {
        let start = CFAbsoluteTimeGetCurrent()
        commands.commit()
        commands.waitUntilCompleted()
        profiler?.record(label, commandBuffer: commands,
                         wall: CFAbsoluteTimeGetCurrent() - start)
    }

    /// Allocates a shared-storage buffer holding `values`.
    ///
    /// Shared storage is the whole point on Apple Silicon: the CPU writes a page
    /// and the GPU reads that same page with no copy.
    public func buffer<T>(_ values: [T]) throws -> MTLBuffer {
        let bytes = MemoryLayout<T>.stride * values.count
        guard let buffer = values.withUnsafeBytes({ raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: bytes, options: .storageModeShared)
        }) else {
            throw MetalError.bufferAllocationFailed(bytes: bytes)
        }
        return buffer
    }

    /// Allocates a zeroed shared-storage buffer for `count` elements of `T`.
    public func emptyBuffer<T>(of type: T.Type, count: Int) throws -> MTLBuffer {
        let bytes = MemoryLayout<T>.stride * count
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed(bytes: bytes)
        }
        return buffer
    }
}
