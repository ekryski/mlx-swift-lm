# Run tests. Requires xcodebuild (Metal shaders are not built by "swift test").
.PHONY: test
test:
	./scripts/test.sh
