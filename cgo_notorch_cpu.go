//go:build !cuda

// CPU-build linkage for the notorch bridge — links the plain libnotorch.a
// (CPU + BLAS, no CUDA). The cuda build links libnotorch_gpu.a instead
// (cgo_notorch_cuda.go). Split out of cgo_notorch.go so the LDFLAGS differ
// by build tag without touching the shared bridge code.

package main

/*
#cgo linux LDFLAGS: -L/usr/local/lib -lnotorch -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas -lm
#cgo darwin LDFLAGS: -L/opt/homebrew/lib -lnotorch
*/
import "C"
