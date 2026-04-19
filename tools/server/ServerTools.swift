import Foundation

// MARK: - Tool Call Parsing

extension SimpleHTTPServer {

    /// Parse ALL tool calls from text. Supports MiniMax XML, generic XML, and <function=> formats.
    func parseAllToolCalls(_ text: String) -> [(name: String, arguments: String)] {
        let minimax = parseAllMiniMaxToolCalls(text)
        if !minimax.isEmpty { return minimax }
        if let tc = parseToolCallXML(text) { return [tc] }
        return []
    }

    /// Parse generic XML tool call: `<function=name><parameter=key>value</parameter></function>`
    func parseToolCallXML(_ text: String) -> (name: String, arguments: String)? {
        let all = parseAllMiniMaxToolCalls(text)
        if let first = all.first { return first }

        guard let funcStart = text.range(of: "<function=") else { return nil }
        guard let nameEnd = text.range(of: ">", range: funcStart.upperBound..<text.endIndex) else { return nil }

        let funcName = String(text[funcStart.upperBound..<nameEnd.lowerBound])

        var args: [String: String] = [:]
        var search = nameEnd.upperBound
        while let paramStart = text.range(of: "<parameter=", range: search..<text.endIndex) {
            guard let pNameEnd = text.range(of: ">", range: paramStart.upperBound..<text.endIndex) else { break }
            let paramName = String(text[paramStart.upperBound..<pNameEnd.lowerBound])
            guard let paramEnd = text.range(of: "</parameter>", range: pNameEnd.upperBound..<text.endIndex) else { break }
            var value = String(text[pNameEnd.upperBound..<paramEnd.lowerBound])
            if value.hasPrefix("\n") { value = String(value.dropFirst()) }
            if value.hasSuffix("\n") { value = String(value.dropLast()) }
            args[paramName] = value
            search = paramEnd.upperBound
        }

        let argsJSON = (try? JSONSerialization.data(withJSONObject: args)) ?? Data()
        return (name: funcName, arguments: String(data: argsJSON, encoding: .utf8) ?? "{}")
    }

    /// Parse ALL `<invoke>` blocks from MiniMax tool call XML.
    func parseAllMiniMaxToolCalls(_ text: String) -> [(name: String, arguments: String)] {
        guard text.contains("<invoke name=") || text.contains("<parameter name=") else { return [] }
        var results: [(name: String, arguments: String)] = []

        var searchStart = text.startIndex
        while let invokeStart = text.range(of: "<invoke name=\"", range: searchStart..<text.endIndex) {
            guard let invokeEnd = text.range(of: "</invoke>", range: invokeStart.upperBound..<text.endIndex) else { break }
            let invokeBlock = String(text[invokeStart.lowerBound..<invokeEnd.upperBound])
            if let tc = parseSingleMiniMaxInvoke(invokeBlock) {
                results.append(tc)
            }
            searchStart = invokeEnd.upperBound
        }

        // Fallback: malformed XML with <parameter> but no <invoke name=">
        if results.isEmpty && text.contains("<parameter name=") && !text.contains("<invoke name=") {
            var args: [String: String] = [:]
            var paramSearch = text.startIndex
            while let paramStart = text.range(of: "<parameter name=\"", range: paramSearch..<text.endIndex) {
                let afterParam = text[paramStart.upperBound...]
                guard let pNameEnd = afterParam.range(of: "\">") else { break }
                let paramName = String(afterParam[afterParam.startIndex..<pNameEnd.lowerBound])
                let valueStart = pNameEnd.upperBound
                guard let paramEnd = text.range(of: "</parameter>", range: valueStart..<text.endIndex) else { break }
                var value = String(text[valueStart..<paramEnd.lowerBound])
                if value.hasPrefix("\n") { value = String(value.dropFirst()) }
                if value.hasSuffix("\n") { value = String(value.dropLast()) }
                args[paramName] = value
                paramSearch = paramEnd.upperBound
            }
            if !args.isEmpty {
                let toolName: String
                if args["command"] != nil { toolName = "terminal" }
                else if args["content"] != nil && args["path"] != nil { toolName = "write_file" }
                else if args["path"] != nil { toolName = "read_file" }
                else if args["query"] != nil { toolName = "grep" }
                else { toolName = "terminal" }
                let argsJSON = (try? JSONSerialization.data(withJSONObject: args)) ?? Data()
                results.append((name: toolName, arguments: String(data: argsJSON, encoding: .utf8) ?? "{}"))
            }
        }

        return results
    }

    func parseSingleMiniMaxInvoke(_ text: String) -> (name: String, arguments: String)? {
        guard let invokeStart = text.range(of: "<invoke name=\"") else { return nil }
        let afterName = text[invokeStart.upperBound...]
        guard let nameEnd = afterName.range(of: "\"") else { return nil }
        let funcName = String(afterName[afterName.startIndex..<nameEnd.lowerBound])

        var args: [String: String] = [:]
        var search = nameEnd.upperBound
        while let paramStart = text.range(of: "<parameter name=\"", range: search..<text.endIndex) {
            let afterParam = text[paramStart.upperBound...]
            guard let pNameEnd = afterParam.range(of: "\">") else { break }
            let paramName = String(afterParam[afterParam.startIndex..<pNameEnd.lowerBound])
            let valueStart = pNameEnd.upperBound
            guard let paramEnd = text.range(of: "</parameter>", range: valueStart..<text.endIndex) else { break }
            var value = String(text[valueStart..<paramEnd.lowerBound])
            if value.hasPrefix("\n") { value = String(value.dropFirst()) }
            if value.hasSuffix("\n") { value = String(value.dropLast()) }
            args[paramName] = value
            search = paramEnd.upperBound
        }

        let argsJSON = (try? JSONSerialization.data(withJSONObject: args)) ?? Data()
        return (name: funcName, arguments: String(data: argsJSON, encoding: .utf8) ?? "{}")
    }
}
