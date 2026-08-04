import Foundation
import Godwit

// Placeholder entry point. The CLI grows once there is a model format to load;
// for now it exists so the package has a runnable product.

let arguments = CommandLine.arguments.dropFirst()

guard let command = arguments.first else {
    print("""
    godwit — streaming mixture-of-experts inference for memory-constrained machines

    usage: godwit <command>

    commands:
      version    print the version
    """)
    exit(0)
}

switch command {
case "version":
    print("godwit 0.0.1-dev")
default:
    FileHandle.standardError.write(Data("unknown command: \(command)\n".utf8))
    exit(1)
}
