// Copyright © 2026 Apple Inc.

import MLXLMCommon
import XCTest

/// These tests intentionally stay policy-math only.
///
/// Rationale:
/// - `mlx-swift` already covers manager/ticket event-stream behavior in depth.
/// - `mlx-swift-lm` should validate only the policy layer it adds on top.
/// - Keeping this target model-free avoids large downloads so tests run in CI.
final class WiredMemoryPolicyTests: XCTestCase {
    func testWiredSumPolicyCapAffectsLimitAndAdmission() {
        let policy = WiredSumPolicy(cap: 200)

        XCTAssertEqual(policy.limit(baseline: 100, activeSizes: [50, 100]), 200)
        XCTAssertTrue(policy.canAdmit(baseline: 100, activeSizes: [50], newSize: 50))
        XCTAssertFalse(policy.canAdmit(baseline: 100, activeSizes: [50], newSize: 51))
    }

    func testWiredMaxPolicyReturnsLargestDemandOrBaseline() {
        let policy = WiredMaxPolicy()

        XCTAssertEqual(policy.limit(baseline: 100, activeSizes: [20, 150, 60]), 150)
        XCTAssertEqual(policy.limit(baseline: 200, activeSizes: [20, 150, 60]), 200)
    }

    func testWiredFixedPolicyIgnoresActiveSizes() {
        let policy = WiredFixedPolicy(limit: 123)

        XCTAssertEqual(policy.limit(baseline: 0, activeSizes: []), 123)
        XCTAssertEqual(policy.limit(baseline: 500, activeSizes: [1, 2, 3]), 123)
    }

    func testWiredBudgetPolicyIdentityAndCapBehavior() {
        let sharedID = UUID()
        let first = WiredBudgetPolicy(baseBytes: 100, cap: 300, id: sharedID)
        let second = WiredBudgetPolicy(baseBytes: 999, cap: 999, id: sharedID)
        let third = WiredBudgetPolicy(baseBytes: 100, cap: 300, id: UUID())

        XCTAssertEqual(first, second)
        XCTAssertNotEqual(first, third)
        XCTAssertEqual(first.limit(baseline: 50, activeSizes: [75]), 225)
        XCTAssertTrue(first.canAdmit(baseline: 50, activeSizes: [75], newSize: 75))
        XCTAssertFalse(first.canAdmit(baseline: 50, activeSizes: [75], newSize: 76))
    }

    // MARK: - WiredMemoryUtils Budget Estimation

    func testTicketFromMeasurementRespectsHeadroom() {
        let measurement = WiredMemoryMeasurement(
            weightBytes: 1_000_000,
            kvBytes: 200_000,
            workspaceBytes: 100_000,
            peakActiveBytes: 1_300_000,
            tokenCount: 1024,
            prefillStepSize: 512
        )

        let ticket10 = WiredMemoryUtils.ticket(from: measurement, headroom: 0.1)
        let ticket20 = WiredMemoryUtils.ticket(from: measurement, headroom: 0.2)

        // Total = 1,300,000. With 10% headroom = 1,430,000. With 20% = 1,560,000.
        XCTAssertEqual(ticket10.size, 1_430_000)
        XCTAssertEqual(ticket20.size, 1_560_000)
    }

    func testMeasurementTotalBytesIsSum() {
        let m = WiredMemoryMeasurement(
            weightBytes: 500,
            kvBytes: 300,
            workspaceBytes: 200,
            peakActiveBytes: 1000,
            tokenCount: 128,
            prefillStepSize: 512
        )
        XCTAssertEqual(m.totalBytes, 1000)
    }

    func testMeasurementNegativeComponentsClamped() {
        let m = WiredMemoryMeasurement(
            weightBytes: 500,
            kvBytes: -100,  // e.g., from measurement noise
            workspaceBytes: -50,
            peakActiveBytes: 400,
            tokenCount: 128,
            prefillStepSize: 512
        )
        // Negative components should be clamped to 0
        XCTAssertEqual(m.totalBytes, 500)  // max(0, 500) + max(0, -100) + max(0, -50) = 500
    }
}
