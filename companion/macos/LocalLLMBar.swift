// LocalLLM Bar: a menu-bar companion for the self-hosted Local LLM server.
//
// Why a native app: the web UI answers "is the model up, is something queued,
// is a node down, is an approval waiting for me" only while a tab is open. This
// puts that answer in the menu bar for the cost of one small request every 5 s
// (GET /api/status/summary, which the server shapes for exactly this caller)
// and offers the few levers an operator reaches for: restart the model, cancel
// running tasks, open the UI, copy the API URL.
//
// Deliberately one file, AppKit plus Foundation plus Security, no third-party
// code and no Xcode project: `deploy.py companion build` compiles it with swiftc.
// The API token lives in the login Keychain; the server URL (not a secret) in
// the app's user defaults. On first run both default from the CLI's credentials
// file (~/.config/local-llm/credentials.json) when it exists; that file is only
// read, never written.
//
// Polling backs off exponentially (5 s up to 60 s) while the server is
// unreachable or refuses the token, so a stopped server or a revoked token does
// not turn into a request every 5 s forever.

import AppKit
import Foundation
import Security

// MARK: - Keychain

enum BarKeychain {
    static let service = "local.llm.bar"
    static let tokenAccount = "api-token"

    private static func baseQuery(_ account: String) -> [String: Any] {
        return [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }

    static func read(_ account: String) -> String? {
        var query = baseQuery(account)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else { return nil }
        let value = String(data: data, encoding: .utf8)
        return (value?.isEmpty ?? true) ? nil : value
    }

    /// Replace (or, with nil/empty, remove) the stored value.
    @discardableResult
    static func write(_ account: String, _ value: String?) -> OSStatus {
        let base = baseQuery(account)
        SecItemDelete(base as CFDictionary)
        guard let value = value, !value.isEmpty else { return errSecSuccess }
        var add = base
        add[kSecValueData as String] = Data(value.utf8)
        // Readable after the first unlock, never synced or migrated to another Mac.
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        add[kSecAttrLabel as String] = "LocalLLM Bar API token"
        return SecItemAdd(add as CFDictionary, nil)
    }
}

// MARK: - Settings

struct BarSettings {
    static let urlKey = "serverURL"
    static let defaultURL = "http://127.0.0.1:8000"

    var serverURL: String
    var token: String?

    static func load() -> BarSettings {
        var url = UserDefaults.standard.string(forKey: urlKey)
        if url?.isEmpty ?? true { url = nil }
        var token = BarKeychain.read(BarKeychain.tokenAccount)
        if url == nil || token == nil {
            let cli = cliCredentials()
            if url == nil { url = cli.url }
            if token == nil { token = cli.token }
        }
        return BarSettings(serverURL: normalise(url ?? defaultURL), token: token)
    }

    /// "host:8000/" -> "http://host:8000": trimmed, a scheme, no trailing slash.
    static func normalise(_ raw: String) -> String {
        var value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.isEmpty { value = defaultURL }
        if !value.lowercased().hasPrefix("http://") && !value.lowercased().hasPrefix("https://") {
            value = "http://" + value
        }
        while value.hasSuffix("/") { value.removeLast() }
        return value
    }

    /// The CLI's saved sign-in, used only as a first-run default.
    ///
    /// Tolerant of both shapes a CLI might write: a flat {"url", "token"} object,
    /// or {"default": url, "servers": {url: {"token": ...}}}.
    static func cliCredentials() -> (url: String?, token: String?) {
        let path = (NSHomeDirectory() as NSString)
            .appendingPathComponent(".config/local-llm/credentials.json")
        guard let data = FileManager.default.contents(atPath: path),
              let object = try? JSONSerialization.jsonObject(with: data),
              let root = object as? [String: Any] else { return (nil, nil) }
        func pick(_ dict: [String: Any], _ keys: [String]) -> String? {
            for key in keys {
                if let value = dict[key] as? String, !value.isEmpty { return value }
            }
            return nil
        }
        let urlKeys = ["url", "server", "base_url", "server_url"]
        let tokenKeys = ["token", "api_token", "access_token"]
        var url = pick(root, urlKeys)
        var token = pick(root, tokenKeys)
        if let servers = root["servers"] as? [String: Any], !servers.isEmpty {
            let chosen = pick(root, ["default", "current"]) ?? url ?? servers.keys.sorted().first
            if let chosen = chosen, let entry = servers[chosen] as? [String: Any] {
                url = pick(entry, urlKeys) ?? chosen
                token = token ?? pick(entry, tokenKeys)
            }
        }
        return (url, token)
    }
}

// MARK: - JSON helpers (the summary is read loosely: any field may be null)

private func jDict(_ value: Any?) -> [String: Any]? { return value as? [String: Any] }
private func jStr(_ value: Any?) -> String? {
    if let s = value as? String { return s.isEmpty ? nil : s }
    if let n = value as? NSNumber { return n.stringValue }
    return nil
}
private func jNum(_ value: Any?) -> Double? { return (value as? NSNumber)?.doubleValue }
private func jInt(_ value: Any?) -> Int? { return (value as? NSNumber)?.intValue }
private func jBool(_ value: Any?) -> Bool { return (value as? NSNumber)?.boolValue ?? false }

enum BarError: LocalizedError {
    case badURL
    case badPayload
    case http(Int, String?)

    var errorDescription: String? {
        switch self {
        case .badURL: return "The server URL is not valid: set it in Settings."
        case .badPayload: return "The server answered with something that is not a status summary."
        case .http(let code, let message): return message ?? "HTTP \(code)"
        }
    }
}

// MARK: - The menu-bar controller

@MainActor
final class BarController: NSObject, NSApplicationDelegate, NSMenuDelegate {
    static let pollInterval: TimeInterval = 5
    static let maxBackoff: TimeInterval = 60

    private var statusItem: NSStatusItem!
    private let menu = NSMenu()
    private var settings = BarSettings.load()
    private var summary: [String: Any]?
    private var problem: String?
    private var lastAction: String?
    private var failures = 0
    private var nextDelay: TimeInterval = BarController.pollInterval
    private var pollTask: Task<Void, Never>?

    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 4
        config.timeoutIntervalForResource = 10
        config.waitsForConnectivity = false
        config.urlCache = nil
        config.httpShouldSetCookies = false
        return URLSession(configuration: config)
    }()

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        installEditMenu()
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.imagePosition = .imageLeft
        menu.autoenablesItems = false
        menu.delegate = self
        statusItem.menu = menu
        show(label: "...", color: "gray", tooltip: "LocalLLM: connecting to \(settings.serverURL)")
        startPolling()
    }

    func applicationWillTerminate(_ notification: Notification) {
        pollTask?.cancel()
    }

    /// An accessory app has no menu bar of its own, so without this Cmd-V does
    /// nothing in the Settings fields (and a token is exactly what gets pasted).
    private func installEditMenu() {
        let main = NSMenu()
        let editItem = NSMenuItem()
        main.addItem(editItem)
        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = edit
        NSApp.mainMenu = main
    }

    // MARK: Polling

    private func startPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self = self else { return }
                let delay = await self.refresh()
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
        }
    }

    private func backoff() -> TimeInterval {
        failures += 1
        let exponent = Double(min(failures, 6) - 1)
        return min(BarController.maxBackoff, BarController.pollInterval * pow(2, exponent))
    }

    private func endpoint(_ path: String) -> URL? {
        return URL(string: settings.serverURL + path)
    }

    private func authorise(_ request: inout URLRequest) {
        if let token = settings.token, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("LocalLLMBar/1.0", forHTTPHeaderField: "User-Agent")
    }

    private func serverMessage(_ data: Data) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: data),
              let dict = object as? [String: Any] else { return nil }
        return jStr(dict["error"]) ?? jStr(dict["detail"])
    }

    /// One poll. Returns the delay before the next one.
    func refresh() async -> TimeInterval {
        guard let url = endpoint("/api/status/summary") else {
            summary = nil
            problem = BarError.badURL.localizedDescription
            show(label: "setup", color: "gray", tooltip: problem ?? "")
            return BarController.maxBackoff
        }
        var request = URLRequest(url: url)
        authorise(&request)
        do {
            let (data, response) = try await session.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            if code == 401 || code == 403 {
                summary = nil
                problem = code == 401
                    ? "The server wants an API token: open Settings."
                    : (serverMessage(data) ?? "This API token lacks the status scope.")
                let delay = backoff()
                show(label: code == 401 ? "sign in" : "no access", color: "gray",
                     tooltip: "LocalLLM: \(problem ?? "")")
                return delay
            }
            guard code == 200 else { throw BarError.http(code, serverMessage(data)) }
            guard let object = try? JSONSerialization.jsonObject(with: data),
                  let dict = object as? [String: Any] else { throw BarError.badPayload }
            summary = dict
            problem = nil
            failures = 0
            let indicator = jDict(dict["indicator"]) ?? [:]
            let label = jStr(indicator["label"]) ?? "unknown"
            show(label: label, color: jStr(indicator["color"]) ?? "gray",
                 tooltip: "LocalLLM: \(label) at \(settings.serverURL)")
            return BarController.pollInterval
        } catch {
            summary = nil
            let delay = backoff()
            problem = "\(error.localizedDescription) (retrying in \(Int(delay)) s)"
            show(label: "offline", color: "red", tooltip: "LocalLLM: cannot reach \(settings.serverURL)")
            return delay
        }
    }

    // MARK: Status item

    private static func colour(_ name: String) -> NSColor {
        switch name {
        case "green": return .systemGreen
        case "yellow": return .systemYellow
        case "orange": return .systemOrange
        case "red": return .systemRed
        default: return .systemGray
        }
    }

    private static func dot(_ colour: NSColor) -> NSImage {
        let image = NSImage(size: NSSize(width: 10, height: 10), flipped: false) { rect in
            colour.setFill()
            NSBezierPath(ovalIn: rect.insetBy(dx: 1, dy: 1)).fill()
            return true
        }
        image.isTemplate = false
        return image
    }

    private func show(label: String, color: String, tooltip: String) {
        guard let button = statusItem?.button else { return }
        button.image = BarController.dot(BarController.colour(color))
        button.title = " " + label
        button.toolTip = tooltip
    }

    // MARK: Menu (built when it opens, from the last poll)

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        let info = { (title: String) in
            let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
            item.isEnabled = false
            menu.addItem(item)
        }
        let action = { (title: String, selector: Selector, enabled: Bool, key: String) in
            let item = NSMenuItem(title: title, action: selector, keyEquivalent: key)
            item.target = self
            item.isEnabled = enabled
            menu.addItem(item)
        }

        var running: [String] = []
        var approvals = 0
        var controls: [String: Any] = [:]
        if let s = summary {
            let model = jDict(s["model"]) ?? [:]
            let name = jStr(model["alias"]) ?? jStr(model["id"]) ?? "unknown model"
            info("Model: \(name) (\(jStr(model["status"]) ?? "unknown"))")
            if let adapter = jStr(model["adapter"]) { info("Adapter: \(adapter)") }
            var backend = "Backend: \(jStr(model["backend"]) ?? "unknown")"
            let caps = (model["capabilities"] as? [Any] ?? []).compactMap { jStr($0) }
            if !caps.isEmpty { backend += " (\(caps.joined(separator: ", ")))" }
            info(backend)
            let queue = jDict(s["queue"]) ?? [:]
            var queueLine = "Queue: \(jInt(queue["running"]) ?? 0) running, \(jInt(queue["waiting"]) ?? 0) waiting"
            if let limit = jInt(queue["limit"]) { queueLine += ", limit \(limit)" }
            info(queueLine)
            if let metrics = jDict(s["metrics"]) {
                var line = "Last hour: \(jInt(metrics["requests"]) ?? 0) requests"
                if let ttft = jNum(metrics["ttft_p50_ms"]) { line += String(format: ", TTFT %.0f ms", ttft) }
                if let tps = jNum(metrics["decode_tps_p50"]) { line += String(format: ", %.1f tok/s", tps) }
                info(line)
            }
            if let nodes = s["nodes"] as? [[String: Any]], nodes.count > 1 {
                let healthy = nodes.filter { jStr($0["state"]) == "healthy" }.count
                info("Nodes: \(healthy) of \(nodes.count) healthy")
                for node in nodes.prefix(8) {
                    let role = jStr(node["role"]).map { ", \($0)" } ?? ""
                    info("    \(jStr(node["name"]) ?? "?"): \(jStr(node["state"]) ?? "unknown")\(role)")
                }
            }
            if let tasks = jDict(s["tasks"]) {
                running = (tasks["running_ids"] as? [Any] ?? []).compactMap { jStr($0) }
                info("Tasks running: \(jInt(tasks["running"]) ?? running.count)")
            }
            if let retrain = jDict(s["retrain"]), jBool(retrain["running"]) {
                info("Retraining: \(jStr(retrain["message"]) ?? "running")")
            }
            approvals = jInt(jDict(s["approvals"])?["pending"]) ?? 0
            controls = jDict(s["controls"]) ?? [:]
        } else {
            info("Server: \(settings.serverURL)")
            info(problem ?? "Connecting...")
        }
        if settings.serverURL.lowercased().hasPrefix("http://") && settings.token != nil
            && !BarController.isLocal(settings.serverURL) {
            info("Warning: the token travels unencrypted (http)")
        }
        if let last = lastAction { info(last) }

        menu.addItem(.separator())
        action("Open Web UI", #selector(openWebUI(_:)), true, "o")
        action(approvals > 0 ? "Approvals Pending: \(approvals)" : "No Approvals Pending",
               #selector(openWebUI(_:)), approvals > 0, "")
        action("Restart Model...", #selector(restartModel(_:)), jBool(controls["restart_model"]), "")
        action(running.isEmpty ? "Cancel Running Tasks" : "Cancel Running Tasks (\(running.count))...",
               #selector(cancelTasks(_:)), jBool(controls["cancel_tasks"]) && !running.isEmpty, "")
        if jBool(controls["retrain"]) {
            action("Retrain Now...", #selector(retrain(_:)), true, "")
        }
        action("Copy API URL", #selector(copyAPIURL(_:)), true, "c")
        action("Refresh Now", #selector(refreshNow(_:)), true, "r")
        menu.addItem(.separator())
        action("Settings...", #selector(openSettings(_:)), true, ",")
        action("Quit LocalLLM Bar", #selector(quit(_:)), true, "q")
    }

    private static func isLocal(_ url: String) -> Bool {
        guard let host = URL(string: url)?.host?.lowercased() else { return false }
        return host == "localhost" || host == "127.0.0.1" || host == "::1" || host.hasSuffix(".local")
    }

    // MARK: Actions

    private func confirm(_ title: String, _ detail: String, _ button: String) -> Bool {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = detail
        alert.alertStyle = .warning
        alert.addButton(withTitle: button)
        alert.addButton(withTitle: "Cancel")
        NSApp.activate(ignoringOtherApps: true)
        return alert.runModal() == .alertFirstButtonReturn
    }

    private func report(_ title: String, _ detail: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = detail
        alert.alertStyle = .critical
        NSApp.activate(ignoringOtherApps: true)
        alert.runModal()
    }

    /// POST to an existing admin route; alerts on failure, returns success.
    @discardableResult
    private func post(_ path: String, what: String, quiet: Bool = false) async -> Bool {
        guard let url = endpoint(path) else {
            report("\(what) failed", BarError.badURL.localizedDescription)
            return false
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        authorise(&request)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("{}".utf8)
        do {
            let (data, response) = try await session.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            if (200..<300).contains(code) { return true }
            if !quiet { report("\(what) failed", serverMessage(data) ?? "HTTP \(code)") }
        } catch {
            if !quiet { report("\(what) failed", error.localizedDescription) }
        }
        return false
    }

    private func stamp() -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        formatter.dateStyle = .none
        return formatter.string(from: Date())
    }

    @objc private func openWebUI(_ sender: Any?) {
        if let url = URL(string: settings.serverURL) { NSWorkspace.shared.open(url) }
    }

    @objc private func restartModel(_ sender: Any?) {
        guard confirm("Restart the model?",
                      "Requests in flight fail and the model reloads, which can take a minute.",
                      "Restart") else { return }
        Task {
            if await post("/api/model/restart", what: "Restart") {
                lastAction = "Restart requested at \(stamp())"
                startPolling()
            }
        }
    }

    @objc private func cancelTasks(_ sender: Any?) {
        let ids = (jDict(summary?["tasks"])?["running_ids"] as? [Any] ?? []).compactMap { jStr($0) }
        guard !ids.isEmpty else { return }
        guard confirm("Cancel \(ids.count) running task\(ids.count == 1 ? "" : "s")?",
                      "Each task stops at its next step; its run is recorded as cancelled.",
                      "Cancel Tasks") else { return }
        Task {
            var allowed = CharacterSet.urlPathAllowed
            allowed.remove(charactersIn: "/")
            var done = 0
            for id in ids {
                let escaped = id.addingPercentEncoding(withAllowedCharacters: allowed) ?? id
                if await post("/api/tasks/\(escaped)/cancel", what: "Cancel task", quiet: done > 0) {
                    done += 1
                }
            }
            lastAction = "Cancelled \(done) of \(ids.count) task(s) at \(stamp())"
            startPolling()
        }
    }

    @objc private func retrain(_ sender: Any?) {
        guard confirm("Start retraining now?",
                      "Retraining takes over the model server until it finishes.",
                      "Retrain") else { return }
        Task {
            if await post("/api/retrain", what: "Retrain") {
                lastAction = "Retrain requested at \(stamp())"
                startPolling()
            }
        }
    }

    @objc private func copyAPIURL(_ sender: Any?) {
        let path = jStr(summary?["api_base_path"]) ?? "/v1"
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(settings.serverURL + path, forType: .string)
    }

    @objc private func refreshNow(_ sender: Any?) {
        failures = 0
        startPolling()
    }

    @objc private func openSettings(_ sender: Any?) {
        let alert = NSAlert()
        alert.messageText = "LocalLLM Bar Settings"
        alert.informativeText = "The server URL, and an API token with the status scope "
            + "(add control to restart the model or cancel tasks). "
            + "The token is kept in your login Keychain."
        let box = NSView(frame: NSRect(x: 0, y: 0, width: 340, height: 104))
        let urlLabel = NSTextField(labelWithString: "Server URL")
        urlLabel.frame = NSRect(x: 0, y: 82, width: 340, height: 18)
        let urlField = NSTextField(frame: NSRect(x: 0, y: 56, width: 340, height: 24))
        urlField.stringValue = settings.serverURL
        urlField.placeholderString = BarSettings.defaultURL
        let tokenLabel = NSTextField(labelWithString: "API token")
        tokenLabel.frame = NSRect(x: 0, y: 28, width: 340, height: 18)
        let tokenField = NSSecureTextField(frame: NSRect(x: 0, y: 2, width: 340, height: 24))
        tokenField.placeholderString = settings.token == nil
            ? "paste a token (leave empty if the server has no sign-in)"
            : "unchanged (a token is stored)"
        for view in [urlLabel, urlField, tokenLabel, tokenField] { box.addSubview(view) }
        alert.accessoryView = box
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: "Forget Token")
        NSApp.activate(ignoringOtherApps: true)
        alert.window.initialFirstResponder = urlField
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            let url = BarSettings.normalise(urlField.stringValue)
            UserDefaults.standard.set(url, forKey: BarSettings.urlKey)
            settings.serverURL = url
            let token = tokenField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            if !token.isEmpty {
                let status = BarKeychain.write(BarKeychain.tokenAccount, token)
                if status != errSecSuccess {
                    report("Could not save the token",
                           (SecCopyErrorMessageString(status, nil) as String?) ?? "Keychain error \(status)")
                }
                settings.token = token
            }
        case .alertThirdButtonReturn:
            BarKeychain.write(BarKeychain.tokenAccount, nil)
            settings.token = nil
        default:
            return
        }
        failures = 0
        lastAction = nil
        startPolling()
    }

    @objc private func quit(_ sender: Any?) {
        NSApp.terminate(nil)
    }
}

// MARK: - Entry point

@main
enum LocalLLMBarMain {
    @MainActor
    static func main() {
        let app = NSApplication.shared
        let controller = BarController()
        app.delegate = controller
        app.setActivationPolicy(.accessory)
        // NSApplication holds its delegate weakly: keep the controller alive
        // for as long as the run loop runs.
        withExtendedLifetime(controller) {
            app.run()
        }
    }
}
