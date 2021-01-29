/* Edge Impulse Arduino examples
 * Copyright (c) 2020 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "numpy_types.h"
#include "ei_classifier_porting.h"
#include "ei_classifier_types.h"

#include <alsa/asoundlib.h>

int microphone_audio_signal_get_data(size_t, size_t, float *);
void microphone_inference_end();
void readData();
int run_classifier(signal_t *, ei_impulse_result_t *, bool);
void le16_to_float(char *, float *, size_t);

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK 0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4

static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal

const int ms1000 = EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(short signed int);
const int ms250 = ms1000 / 4;

char alsaBuffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(short signed int) * 3];   // 3 seconds
char classifierBuffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(short signed int)]; // 1 second

snd_pcm_t *capture_handle;
int channels = 1;
unsigned int rate = EI_CLASSIFIER_FREQUENCY;
snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;
char *card = "hw:1";

void *initAlsa()
{

    int err;

    snd_pcm_hw_params_t *hw_params;

    if ((err = snd_pcm_open(&capture_handle, card, SND_PCM_STREAM_CAPTURE, 0)) < 0)
    {
        fprintf(stderr, "cannot open audio device %s (%s)\n",
                card,
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "audio interface opened\n");

    if ((err = snd_pcm_hw_params_malloc(&hw_params)) < 0)
    {
        fprintf(stderr, "cannot allocate hardware parameter structure (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "hw_params allocated\n");

    if ((err = snd_pcm_hw_params_any(capture_handle, hw_params)) < 0)
    {
        fprintf(stderr, "cannot initialize hardware parameter structure (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "hw_params initialized\n");

    if ((err = snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0)
    {
        fprintf(stderr, "cannot set access type (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "hw_params access set\n");

    if ((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, format)) < 0)
    {
        fprintf(stderr, "cannot set sample format (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "hw_params format set\n");

    if ((err = snd_pcm_hw_params_set_rate(capture_handle, hw_params, rate, 0)) < 0)
    {
        fprintf(stderr, "cannot set sample rate (%s)\n",
                snd_strerror(err));
        exit(1);
    }
    else
    {
        unsigned int read_rate;
        int read_dir;

        snd_pcm_hw_params_get_rate(hw_params, &read_rate, &read_dir);

        fprintf(stdout, "hw_params rate set: %d\n", read_rate);
    }

    if ((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, channels)) < 0)
    {
        fprintf(stderr, "cannot set channel count (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "hw_params channels set:%d\n", channels);

    if ((err = snd_pcm_hw_params(capture_handle, hw_params)) < 0)
    {
        fprintf(stderr, "cannot set parameters (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "hw_params set\n");

    snd_pcm_hw_params_free(hw_params);

    fprintf(stdout, "hw_params freed\n");

    if ((err = snd_pcm_prepare(capture_handle)) < 0)
    {
        fprintf(stderr, "cannot prepare audio interface for use (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "audio interface prepared\n");

    return capture_handle;
}

// void setup()
// {
//     // summary of inferencing settings (from model_metadata.h)
//     // printf("Inferencing settings:\n");
//     // printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
//     // printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
//     // printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
//     // printf("\tNo. of classes: %lu\n", (sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0])));
// }

FILE *fptr;

/**
 * @brief      main function. Runs the inferencing loop.
 */
int main()
{

    // fptr = fopen("sample.txt", "w");
    // if (fptr == NULL)
    // {
    //     printf("Error!");
    //     exit(1);
    // }

    initAlsa();

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    int alsaBufferIdx = 0;
    int classifierBufferIdx = 0;

    // Fill 1s in the buffer.
    for (; alsaBufferIdx < ms1000; alsaBufferIdx += ms250)
    {
        readData(alsaBufferIdx);
    }

    for (int i = 0; i < 100; i++)
    {
        if (alsaBufferIdx >= sizeof(alsaBuffer))
        {
            alsaBufferIdx = 0;
        }
        readData(alsaBufferIdx);
        alsaBufferIdx += ms250;

        if (classifierBufferIdx > sizeof(alsaBuffer) - ms1000)
        {
            classifierBufferIdx = 0;
        }
        memcpy(&classifierBuffer, &alsaBuffer[0] + classifierBufferIdx, ms1000);
        classifierBufferIdx += ms250;

        EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
        if (r != EI_IMPULSE_OK)
        {
            printf("ERR: Failed to run classifier (%d)\n", r);
            exit(1);
        }

        //         printf("[");
        //         for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
        //         {
        //             printf("%s: %.5f", result.classification[ix].label, result.classification[ix].value);
        // #if EI_CLASSIFIER_HAS_ANOMALY == 1
        //             printf(", ");
        // #else
        //             if (ix != EI_CLASSIFIER_LABEL_COUNT - 1)
        //             {
        //                 printf(", ");
        //             }
        // #endif
        //         }
        //         printf("]\n");

        int id = 1;
        if (result.classification[id].value > 0.20)
        {
            printf("            %s: %.5f\n", result.classification[id].label,
                   result.classification[id].value);
        }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
        printf("    anomaly score: %.3f\n", result.anomaly);
#endif
    }

    // fclose(fptr);

    microphone_inference_end();
}

void readData(int index)
{
    int err;

    int bitsPerSample = snd_pcm_format_physical_width(format); // for SND_PCM_FORMAT_S16_LE it is 16 bits or 2 bytes
    int bytesPerSample = bitsPerSample / 8;                    // Convert to Bytes = 2
    int bytesPerFrame = bytesPerSample * channels;
    int buffer_frames = ms250 / bytesPerFrame;

    char *idx = &alsaBuffer[0] + index;
    if ((err = snd_pcm_readi(capture_handle, idx, buffer_frames)) != buffer_frames)
    {
        fprintf(stderr, "read from audio interface failed:%s\n", snd_strerror(err));
        exit(1);
    }

    // for (size_t ix = 0; ix < sizeof(alsaBuffer); ix += 2)
    // {
    //     int x = (int)((short *)(&alsaBuffer[ix]))[0];
    //     fprintf(fptr, "%d,", x);
    //     fprintf(fptr, "%2f,", (float)(x) / 32768);
    // }
}

int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    // offset * 2 because each sample is constructed from  2 bytes hence the array is 2 times the samples count.
    le16_to_float(&classifierBuffer[offset * 2], out_ptr, length);

    return 0;
}

void le16_to_float(char *input, float *output, size_t length)
{
    int ii = 0;
    // 2 bytes per sample are used for the input so need to loop in pairs.
    for (size_t ix = 0; ix < length * 2; ix += 2)
    {
        int x = (int)((short *)(&input[ix]))[0];
        output[ii] = (float)(x) / 32768;
        ii++;

        // fprintf(fptr, "%d,", x);
        // fprintf(fptr, "%2f,", (float)(x) / 32768);
    }
}

void microphone_inference_end()
{
    snd_pcm_drop(capture_handle);
    snd_pcm_close(capture_handle);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
