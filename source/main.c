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
int run_classifier_continuous(signal_t*, ei_impulse_result_t* , bool);
void int16_to_float(short int*, float *, size_t );


// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK 0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 3

/** Audio buffers, pointers and selectors */
typedef struct
{
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

snd_pcm_t *capture_handle;

void *initAlsa()
{
    int channels = 1;
    char *card = "hw:1";
    int err;
    unsigned int rate = 44100;

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

    if ((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, SND_PCM_FORMAT_S16_LE)) < 0)
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

/**
 * @brief      Arduino setup function
 */
void setup()
{
    initAlsa();
    // summary of inferencing settings (from model_metadata.h)
    printf("Inferencing settings:\n");
    printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    printf("\tNo. of classes: %lu\n", (sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0])));

    run_classifier_init();
    if (microphone_inference_start() == false)
    {
        printf("ERR: Failed to setup audio sampling\r\n");
        return;
    }
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

    for (int i = 0; i < 5; i++)
    {
        readData();
        EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
                                
        if (r != EI_IMPULSE_OK)
        {
            printf("ERR: Failed to run classifier (%d)\n", r);
            exit(1);
        }

        if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW))
        {
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
#if EI_CLASSIFIER_HAS_ANOMALY == 1
            printf("    anomaly score: %.3f\n", result.anomaly);
#endif

            print_results = 0;
        }
    }

    microphone_inference_end();
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
void readData()
{
    int err;
    int buffer_frames = 128;

    if ((err = snd_pcm_readi(capture_handle, sampleBuffer, buffer_frames)) != buffer_frames)
    {
        fprintf(stderr, "read from audio interface failed (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    // // read into the sample buffer
    // int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready == true)
    {
        for (int i = 0; i > 1; i++)
        {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples)
            {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

signed short n_samples = EI_CLASSIFIER_SLICE_SIZE;
/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
bool microphone_inference_start()
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL)
    {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL)
    {
        free(inference.buffers[0]);
        return false;
    }

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));

    if (sampleBuffer == NULL)
    {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    // configure the data receive callback
    // PDM.onReceive(&readData);

    // // optionally set the gain, defaults to 20
    // PDM.setGain(80);

    // PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));

    // // initialize PDM with:
    // // - one channel (mono mode)
    // // - a 16 kHz sample rate
    // if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY))
    // {
    //     printf("Failed to start PDM!");
    // }

    record_ready = true;

    return true;
}

void int16_to_float(short int *input, float *output, size_t length)
{
    for (size_t ix = 0; ix < length; ix++)
    {
        output[ix] = (float)(input[ix]) / 32768;
    }
}

/**
 * Get raw audio signal data
 */
int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
void microphone_inference_end()
{
    // PDM.end();
    snd_pcm_drop(capture_handle);
    snd_pcm_close(capture_handle);
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
