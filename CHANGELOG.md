# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0](https://github.com/fabiocicerchia/llm-fit/compare/v1.4.0...v1.5.0) (2026-09-09)


### Features

* **packaging:** man page, OS packages and a staged install ([#59](https://github.com/fabiocicerchia/llm-fit/issues/59)) ([ca33c8e](https://github.com/fabiocicerchia/llm-fit/commit/ca33c8e7139fa39c245ce9430172d3a57a99e957))

## [1.4.0](https://github.com/fabiocicerchia/llm-fit/compare/v1.3.1...v1.4.0) (2026-09-08)


### Features

* add the eight-verb repo contract ([#52](https://github.com/fabiocicerchia/llm-fit/issues/52)) ([ccb4508](https://github.com/fabiocicerchia/llm-fit/commit/ccb4508275ecd92297f3c3fca8aa406a00bbbd00))

## [1.3.1](https://github.com/fabiocicerchia/llm-fit/compare/v1.3.0...v1.3.1) (2026-09-04)

### Bug Fixes

- **ci:** pin the editorconfig-checker binary version ([#47](https://github.com/fabiocicerchia/llm-fit/issues/47)) ([d900ec2](https://github.com/fabiocicerchia/llm-fit/commit/d900ec2f1c0f73ee3fbcb49a7cb0c129408d91c2))

## [1.3.0](https://github.com/fabiocicerchia/llm-fit/compare/v1.2.1...v1.3.0) (2026-09-03)

### Features

- tensor parallelism, speculative decoding, and an accuracy claim that is actually measured ([#44](https://github.com/fabiocicerchia/llm-fit/issues/44)) ([b786302](https://github.com/fabiocicerchia/llm-fit/commit/b786302264e68c72278f45da8d185a4a7cbde4db))

## [1.2.1](https://github.com/fabiocicerchia/llm-fit/compare/v1.2.0...v1.2.1) (2026-08-29)

### Bug Fixes

- **gguf:** refuse a hostile header instead of computing a wrong model size ([#35](https://github.com/fabiocicerchia/llm-fit/issues/35)) ([a4fa6e2](https://github.com/fabiocicerchia/llm-fit/commit/a4fa6e249deae92e5ad3d9dfe12519d9eb273a47))
- unblock quality and clear the Scorecard pinned-dependencies finding ([#37](https://github.com/fabiocicerchia/llm-fit/issues/37)) ([f143689](https://github.com/fabiocicerchia/llm-fit/commit/f14368974901f25e8a155d092d8adb002a243fb8))

## [1.2.0](https://github.com/fabiocicerchia/llm-fit/compare/v1.1.0...v1.2.0) (2026-08-25)

### Features

- **docs:** build the docs site in Actions and drop Read the Docs ([#33](https://github.com/fabiocicerchia/llm-fit/issues/33)) ([da6eaf6](https://github.com/fabiocicerchia/llm-fit/commit/da6eaf66546517a4f0c6847461cc0bfe23083aee))

## [1.1.0](https://github.com/fabiocicerchia/llm-fit/compare/v1.0.1...v1.1.0) (2026-08-24)

### Features

- **gguf:** read the model shape from a local GGUF file ([#23](https://github.com/fabiocicerchia/llm-fit/issues/23)) ([cc41f21](https://github.com/fabiocicerchia/llm-fit/commit/cc41f21e999e8a52bc745cb02d0041e1739ae406))

## [1.0.1](https://github.com/fabiocicerchia/llm-fit/compare/v1.0.0...v1.0.1) (2026-08-13)

### Bug Fixes

- security and code-quality findings ([#19](https://github.com/fabiocicerchia/llm-fit/issues/19)) ([cbe27b9](https://github.com/fabiocicerchia/llm-fit/commit/cbe27b90186f75b9d8a20af97cfd19952d97d914))

## 1.0.0 (2026-08-06)

### Bug Fixes

- **ci:** stop security workflows failing on private repos ([#4](https://github.com/fabiocicerchia/llm-fit/issues/4)) ([85b13a5](https://github.com/fabiocicerchia/llm-fit/commit/85b13a55997bda51ce36d3ea0806d9ca697753ce))
- drop the trailing blank line from .gitignore ([0867bad](https://github.com/fabiocicerchia/llm-fit/commit/0867bad12437b390c506c18cc8186ca84e909a0f))
- **lint:** check deferred Close errors and drop a redundant type ([78573d0](https://github.com/fabiocicerchia/llm-fit/commit/78573d07e09870372da01cab45b5254350d5bf56))
- point the Go module path at this repository ([ead96fb](https://github.com/fabiocicerchia/llm-fit/commit/ead96fbe7774d5d9deeb018ea1b7472d1b30b172))
- **pre-commit:** stop check-yaml failing on Helm templates and multi-doc manifests ([313a50f](https://github.com/fabiocicerchia/llm-fit/commit/313a50fc5fa5bec6ef927d85437fe58ff60a0370))
- resolve gandalf findings and wire Go into CI ([080ba2e](https://github.com/fabiocicerchia/llm-fit/commit/080ba2ecc94d76bdbd4c2b1d9664f0ec456a3d1f))
- spell "unparsable" the way the typos linter expects ([d9c7ff7](https://github.com/fabiocicerchia/llm-fit/commit/d9c7ff77103308340fb4bf58113ddeefbafaaae6))

## [Unreleased]

### Added

- Initial implementation. Not yet released; see the Status section of the
  README for what is verified and what is not.
