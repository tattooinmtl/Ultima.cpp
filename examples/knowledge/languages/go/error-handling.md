---
id: go-error-handling
title: Go Error Handling
type: pattern
tags: [go, errors]
links: []
---
# Error Handling in Go

Errors are values. Return them: `func f() (Result, error)`. Don't panic in libraries. Wrap with context using `fmt.Errorf("doing X: %w", err)`. Check with `errors.Is` / `errors.As`.