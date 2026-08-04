import Darwin
import Foundation

/// A small HTTP/1.1 server built on BSD sockets.
///
/// Deliberately dependency-free. The package pulls in nothing at all today, and
/// a dashboard is not a good reason to start — this needs to serve one page and
/// stream one event source, which is a few hundred lines of sockets.
///
/// Loopback only, and no authentication. It hands an unauthenticated caller a
/// local model; that is fine on `127.0.0.1` and would not be anywhere else.
public final class HTTPServer {
    public struct Request {
        public let method: String
        public let path: String
        public let query: [String: String]
        public let body: Data
    }

    /// A response that streams as it is produced, for token-by-token output.
    public final class EventStream {
        private let socket: Int32
        private let lock = NSLock()
        public private(set) var isOpen = true

        init(socket: Int32) { self.socket = socket }

        /// Sends one server-sent event. Returns false once the peer is gone.
        @discardableResult
        public func send(event: String, json: String) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            guard isOpen else { return false }
            let frame = "event: \(event)\ndata: \(json)\n\n"
            return frame.withCString { pointer -> Bool in
                var written = 0
                let total = strlen(pointer)
                while written < total {
                    // SIGPIPE would kill the process when a browser tab closes.
                    let sent = Darwin.send(socket, pointer + written, total - written, 0)
                    if sent <= 0 { isOpen = false; return false }
                    written += sent
                }
                return true
            }
        }

        func close() {
            lock.lock()
            isOpen = false
            lock.unlock()
        }
    }

    public enum Response {
        case ok(contentType: String, body: Data)
        case json(String)
        case notFound
        case stream((EventStream) -> Void)
    }

    private let port: UInt16
    private var listener: Int32 = -1
    private let handler: (Request) -> Response

    public init(port: UInt16, handler: @escaping (Request) -> Response) {
        self.port = port
        self.handler = handler
    }

    public func start() throws {
        listener = socket(AF_INET, SOCK_STREAM, 0)
        guard listener >= 0 else { throw ServerError.socketFailed(errno) }

        var yes: Int32 = 1
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        // Writing to a closed socket must return an error, not signal us.
        setsockopt(listener, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))

        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        address.sin_addr.s_addr = INADDR_LOOPBACK.bigEndian   // loopback only
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)

        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(listener, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { close(listener); throw ServerError.bindFailed(port, errno) }
        guard Darwin.listen(listener, 16) == 0 else {
            close(listener)
            throw ServerError.listenFailed(errno)
        }
    }

    /// Accepts connections until the process ends.
    public func run() {
        while true {
            let client = accept(listener, nil, nil)
            guard client >= 0 else { continue }
            var yes: Int32 = 1
            setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &yes,
                       socklen_t(MemoryLayout<Int32>.size))
            // Thread per connection. At one browser tab this is not the place
            // to be clever.
            Thread { [weak self] in
                self?.serve(client)
                close(client)
            }.start()
        }
    }

    private func serve(_ client: Int32) {
        guard let request = readRequest(client) else { return }

        switch handler(request) {
        case .notFound:
            write(client, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n"
                  + "Connection: close\r\n\r\n")
        case .json(let body):
            respond(client, contentType: "application/json", body: Data(body.utf8))
        case .ok(let contentType, let body):
            respond(client, contentType: contentType, body: body)
        case .stream(let produce):
            write(client, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                  + "Cache-Control: no-cache\r\nConnection: keep-alive\r\n"
                  + "Access-Control-Allow-Origin: *\r\n\r\n")
            let stream = EventStream(socket: client)
            produce(stream)
            stream.close()
        }
    }

    private func respond(_ client: Int32, contentType: String, body: Data) {
        var header = "HTTP/1.1 200 OK\r\nContent-Type: \(contentType)\r\n"
        header += "Content-Length: \(body.count)\r\n"
        header += "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
        write(client, header)
        body.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            var sent = 0
            while sent < raw.count {
                let n = Darwin.send(client, base + sent, raw.count - sent, 0)
                if n <= 0 { return }
                sent += n
            }
        }
    }

    private func write(_ client: Int32, _ text: String) {
        text.withCString { pointer in
            var sent = 0
            let total = strlen(pointer)
            while sent < total {
                let n = Darwin.send(client, pointer + sent, total - sent, 0)
                if n <= 0 { return }
                sent += n
            }
        }
    }

    private func readRequest(_ client: Int32) -> Request? {
        var buffer = [UInt8](repeating: 0, count: 8192)
        var accumulated = Data()
        var headerEnd: Range<Data.Index>?

        // Headers first, then however much body Content-Length promises.
        while headerEnd == nil {
            let n = recv(client, &buffer, buffer.count, 0)
            if n <= 0 { return nil }
            accumulated.append(contentsOf: buffer[0..<n])
            headerEnd = accumulated.range(of: Data("\r\n\r\n".utf8))
            if accumulated.count > 1 << 20 { return nil }
        }
        guard let separator = headerEnd,
              let headerText = String(data: accumulated[..<separator.lowerBound], encoding: .utf8)
        else { return nil }

        let lines = headerText.components(separatedBy: "\r\n")
        let parts = lines[0].split(separator: " ")
        guard parts.count >= 2 else { return nil }

        var contentLength = 0
        for line in lines.dropFirst() where line.lowercased().hasPrefix("content-length:") {
            contentLength = Int(line.split(separator: ":")[1]
                .trimmingCharacters(in: .whitespaces)) ?? 0
        }

        var body = Data(accumulated[separator.upperBound...])
        while body.count < contentLength {
            let n = recv(client, &buffer, buffer.count, 0)
            if n <= 0 { break }
            body.append(contentsOf: buffer[0..<n])
        }

        let target = String(parts[1])
        let pieces = target.split(separator: "?", maxSplits: 1)
        var query: [String: String] = [:]
        if pieces.count == 2 {
            for pair in pieces[1].split(separator: "&") {
                let kv = pair.split(separator: "=", maxSplits: 1)
                if kv.count == 2 {
                    query[String(kv[0])] = String(kv[1])
                        .replacingOccurrences(of: "+", with: " ").removingPercentEncoding
                        ?? String(kv[1])
                }
            }
        }
        return Request(method: String(parts[0]), path: String(pieces[0]),
                       query: query, body: body)
    }
}

public enum ServerError: Error, CustomStringConvertible {
    case socketFailed(Int32)
    case bindFailed(UInt16, Int32)
    case listenFailed(Int32)

    public var description: String {
        switch self {
        case .socketFailed(let code):
            return "socket() failed: \(String(cString: strerror(code)))"
        case .bindFailed(let port, let code):
            return "could not bind port \(port): \(String(cString: strerror(code)))"
        case .listenFailed(let code):
            return "listen() failed: \(String(cString: strerror(code)))"
        }
    }
}
