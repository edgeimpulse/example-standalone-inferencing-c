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
void run_classifier_init();
bool microphone_inference_start();
void microphone_inference_end();
void readData();
int run_classifier_continuous(signal_t *, ei_impulse_result_t *, bool);
void le16_to_float(short int *, float *, size_t);

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK 0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4

static signed short *sampleBuffer;
signed short n_samples = EI_CLASSIFIER_SLICE_SIZE * sizeof(signed short); // The data returned from ALSA uses 2 bytes to define a sample
static bool debug_nn = false;                                             // Set this to true to see e.g. features generated from the raw signal

snd_pcm_t *capture_handle;
int channels = 1;
unsigned int rate = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;

void *
initAlsa()
{

    char *card = "hw:1";
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

    fprintf(stdout, "hw_params access setted\n");

    if ((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, format)) < 0)
    {
        fprintf(stderr, "cannot set sample format (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "hw_params format setted\n");

    if ((err = snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, 0)) < 0)
    {
        fprintf(stderr, "cannot set sample rate (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "hw_params rate setted\n");
    printf("channels - %d \n", channels);

    if ((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, channels)) < 0)
    {
        fprintf(stderr, "cannot set channel count (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "hw_params channels setted\n");

    if ((err = snd_pcm_hw_params(capture_handle, hw_params)) < 0)
    {
        fprintf(stderr, "cannot set parameters (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    fprintf(stdout, "hw_params setted\n");

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

void setup()
{
    sampleBuffer = (signed short *)malloc((n_samples));
    initAlsa();
    // summary of inferencing settings (from model_metadata.h)
    printf("Inferencing settings:\n");
    printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    printf("\tNo. of classes: %lu\n", (sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0])));

    run_classifier_init();
}

/**
 * @brief      main function. Runs the inferencing loop.
 */
int main()
{
    setup();

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    for (int i = 0; i < 100; i++)
    {
        EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);

        if (r != EI_IMPULSE_OK)
        {
            printf("ERR: Failed to run classifier (%d)\n", r);
            exit(1);
        }

        // print the predictions
        printf("Predictions ");
        printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
               result.timing.dsp, result.timing.classification, result.timing.anomaly);
        printf(": \n");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
        {
            printf("    %s: %.5f\n", result.classification[ix].label,
                   result.classification[ix].value);
        }

        int id = 0;
        if (result.classification[id].value > 0.3)
        {
            printf("Match!!!    %s: %.5f\n", result.classification[id].label,
                   result.classification[id].value);
        }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
        printf("    anomaly score: %.3f\n", result.anomaly);
#endif
    }

    microphone_inference_end();
}

void readData()
{
    int err;

    int bitsPerSample = snd_pcm_format_physical_width(format); // for SND_PCM_FORMAT_S16_LE it is 16 bits or 2 bytes
    int bytesPerSample = bitsPerSample / 8;                    // Convert to Bytes = 2
    int bytesPerFrame = bytesPerSample * channels;
    int buffer_frames = n_samples / bytesPerFrame;

    if ((err = snd_pcm_readi(capture_handle, sampleBuffer, buffer_frames)) != buffer_frames)
    {
        fprintf(stderr, "read from audio interface failed:%s\n", snd_strerror(err));
        exit(1);
    }
}

void le16_to_float(short int *input, float *output, size_t length)
{
    int ii = 0;
    // 2 bytes per sample are used for the input so need to loop in pairs.
    for (size_t ix = 0; ix < length * 2; ix += 2)
    {
        int16_t i = (uint16_t)input[ix] | ((uint16_t)input[ix + 1] << 8);

        float f = (float)(i) / 32768;
        output[ii] = f;
        ii++;
    }
}

int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    readData();
    // printf("offset %d \n", offset);
    le16_to_float(&sampleBuffer[offset], out_ptr, length);

    return 0;
}

void microphone_inference_end()
{
    snd_pcm_drop(capture_handle);
    snd_pcm_close(capture_handle);
    free(sampleBuffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
