// Copyright (c) The Arribada initiative.
// Licensed under the MIT License.
package main

/*
#cgo CFLAGS: -Wall -g -DTF_LITE_DISABLE_X86_NEON=1 -Wno-strict-aliasing
#cgo CFLAGS: -DEIDSP_SIGNAL_C_FN_POINTER=1 -DEI_C_LINKAGE=1
#cgo CFLAGS: -I.
#cgo CFLAGS: -Iedge-impulse-sdk/
#cgo CFLAGS: -Iedge-impulse-sdk/tensorflow
#cgo CFLAGS: -Iedge-impulse-sdk/third_party
#cgo CFLAGS: -Iedge-impulse-sdk/third_party/flatbuffers
#cgo CFLAGS: -Iedge-impulse-sdk/third_party/flatbuffers/include
#cgo CFLAGS: -Iedge-impulse-sdk/third_party/flatbuffers/include/flatbuffers
#cgo CFLAGS: -Iedge-impulse-sdk/third_party/gemmlowp/
#cgo CFLAGS: -Iedge-impulse-sdk/third_party/gemmlowp/fixedpoint
#cgo CFLAGS: -Iedge-impulse-sdk/third_party/gemmlowp/internal
#cgo CFLAGS: -Iedge-impulse-sdk/third_party/ruy
#cgo CFLAGS: -Imodel-parameters
#cgo CFLAGS: -Itflite-model
#cgo CFLAGS: -Iedge-impulse-sdk/anomaly
#cgo CFLAGS: -Iedge-impulse-sdk/classifier
#cgo CFLAGS: -Iedge-impulse-sdk/dsp
#cgo CFLAGS: -Iedge-impulse-sdk/dsp/kissfft
#cgo CFLAGS: -Iedge-impulse-sdk/porting
#cgo CFLAGS: -Iedge-impulse-sdk/CMSIS/Core/Include/
#cgo CFLAGS: -Iedge-impulse-sdk/CMSIS/DSP/Include/
#cgo CFLAGS: -Os
#cgo CFLAGS: -DNDEBUG
#cgo CFLAGS: -g
#cgo CFLAGS: -fPIC

#cgo CXXFLAGS: -std=c++11
#cgo LDFLAGS: -lm -lstdc++ -ldl -lpthread -lasound
#cgo LDFLAGS:-Lbuild/ -llibedgeimpulses


#include <stdio.h>
#include "main.h"

extern void keyword_detected_callback();
*/
import "C"
import "fmt"

const ComponentName = "audio controller"

type Config struct {
	LogLevel string
	Enabled  bool
}

var DefaultConfig = Config{
	LogLevel: "info",
}

//export keyword_detected_callback
func keyword_detected_callback() {
	fmt.Println("called")
}

func main() {
	C.ACFunction()
}
