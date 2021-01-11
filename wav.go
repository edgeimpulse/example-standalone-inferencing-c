package main

import (
	"bufio"
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"time"
	"unsafe"
)

var wf = File{
	SampleRate:      16000,
	Channels:        1,
	SignificantBits: 16,
}

func main() {
	f, err := os.Open("sample.txt")

	if err != nil {
		log.Fatal(err)
	}
	r := csv.NewReader(f)

	records, err := r.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	wav, err := os.Create("audio.wav")
	if err != nil {
		log.Fatal(err)
	}
	wr, err := wf.NewWriter(wav)
	if err != nil {
		log.Fatal(err)
	}
	defer wr.Close()

	for _, record := range records[0] {
		s, err := strconv.ParseFloat(record, 32)
		if err != nil {
			log.Fatal(err)
		}

		bs := make([]byte, 2)

		binary.LittleEndian.PutUint16(bs, uint16(s*32768))
		// order.PutUint16(bs, uint16(s))
		// binary.Write(bs, binary.LittleEndian, myInt)
		// binary.LittleEndian.PutUint16(bs)
		// fmt.Println(bs)
		if err := wr.WriteSample(bs); err != nil {
			log.Fatal(err)
		}
	}

}

func floatToBytes(f float64, nBytes int) []byte {
	bits := math.Float64bits(f)
	bs := make([]byte, 8)
	binary.LittleEndian.PutUint64(bs, bits)
	// trim padding
	switch nBytes {
	case 2:
		return bs[:2]
	case 4:
		return bs[:4]
	}
	return bs
}

type output interface {
	io.Writer
	io.Seeker
	io.Closer
}

// Writer encapsulates a io.WriteSeeker and supplies Functions for writing samples
type Writer struct {
	output
	options   File
	sampleBuf *bufio.Writer

	bytesWritten       int
	sampleBytesWritten int
}

// NewWriter creates a new WaveWriter and writes the header to it
func (file File) NewWriter(out output) (wr *Writer, err error) {
	if file.Channels != 1 {
		err = fmt.Errorf("sorry, only mono currently")
		return
	}

	wr = &Writer{}
	wr.output = out
	wr.sampleBuf = bufio.NewWriter(out)
	wr.options = file

	_, err = wr.Seek(12, os.SEEK_SET)
	if err != nil {
		return
	}

	n, err := wr.output.Write(tokenChunkFmt[:])
	if err != nil {
		return
	}
	wr.bytesWritten += n

	chunkFmt := riffChunkFmt{
		LengthOfHeader: 16,
		AudioFormat:    1,
		NumChannels:    file.Channels,
		SampleRate:     file.SampleRate,
		BytesPerSec:    uint32(file.Channels) * file.SampleRate * (uint32(file.SignificantBits) / 8),
		BytesPerBloc:   (file.SignificantBits / 8) * file.Channels,
		BitsPerSample:  file.SignificantBits,
	}

	err = binary.Write(wr.output, binary.LittleEndian, chunkFmt)
	if err != nil {
		return
	}

	var r riffChunkFmt
	wr.bytesWritten += int(unsafe.Sizeof(r))

	n, err = wr.output.Write(tokenData[:])
	if err != nil {
		return
	}
	wr.bytesWritten += n

	// leave space for the data size
	_, err = wr.Seek(4, os.SEEK_CUR)
	if err != nil {
		return
	}

	return
}

// WriteInt32 writes the sample to the file using the binary package
func (w *Writer) WriteInt32(sample int32) error {
	err := binary.Write(w.sampleBuf, binary.LittleEndian, sample)
	if err != nil {
		return err
	}

	w.sampleBytesWritten += 4

	return err
}

// WriteSample writes a []byte array to file without conversion
func (w *Writer) WriteSample(sample []byte) error {
	if len(sample)*8 != int(w.options.SignificantBits) {
		return fmt.Errorf("incorrect Sample Length %d", len(sample))
	}

	n, err := w.sampleBuf.Write(sample)
	if err != nil {
		return err
	}

	w.sampleBytesWritten += n

	return nil
}

func (w *Writer) Write(data []byte) (int, error) {
	n, err := w.output.Write(data)
	w.sampleBytesWritten += n
	return n, err
}

// Close corrects the filesize information in the header
func (w *Writer) Close() error {
	if err := w.sampleBuf.Flush(); err != nil {
		return err
	}

	_, err := w.Seek(0, os.SEEK_SET)
	if err != nil {
		return err
	}

	header := riffHeader{
		ChunkSize: uint32(w.bytesWritten + 36),
	}
	copy(header.Ftype[:], tokenRiff[:])
	copy(header.ChunkFormat[:], tokenWaveFormat[:])

	err = binary.Write(w.output, binary.LittleEndian, header)
	if err != nil {
		return err
	}

	// write data chunk size
	_, err = w.Seek(0x28, os.SEEK_SET)
	if err != nil {
		return err
	}

	// write chunk size
	err = binary.Write(w.output, binary.LittleEndian, int32(w.sampleBytesWritten))
	if err != nil {
		return err
	}

	return w.output.Close()
}

const (
	maxSize = 2 << 31
)

var (
	tokenRiff       = [4]byte{'R', 'I', 'F', 'F'}
	tokenWaveFormat = [4]byte{'W', 'A', 'V', 'E'}
	tokenChunkFmt   = [4]byte{'f', 'm', 't', ' '}
	tokenData       = [4]byte{'d', 'a', 't', 'a'}
)

// File describes the WAV file
type File struct {
	SampleRate      uint32
	SignificantBits uint16
	Channels        uint16
	NumberOfSamples uint32
	Duration        time.Duration
	AudioFormat     uint16
	SoundSize       uint32
	Canonical       bool
	BytesPerSecond  uint32
}

// 12 byte header
type riffHeader struct {
	Ftype       [4]byte
	ChunkSize   uint32
	ChunkFormat [4]byte
}

// 20
type riffChunkFmt struct {
	LengthOfHeader uint32
	AudioFormat    uint16 // 1 = PCM not compressed
	NumChannels    uint16
	SampleRate     uint32
	BytesPerSec    uint32
	BytesPerBloc   uint16
	BitsPerSample  uint16
}
