module github.com/fabiocicerchia/llm-fit

go 1.24

// Not a language-version bump: 1.24 stays the floor. This pins the *build*
// toolchain to the first release with no known stdlib vulnerabilities
// (crypto/tls GO-2026-5856 and 12 others all land at or below 1.26.5).
toolchain go1.26.5
