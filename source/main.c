#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "numpy_types.h"
#include "ei_classifier_porting.h"
#include "ei_classifier_types.h"
#include <alsa/asoundlib.h>

// static float features[1 * 1024 * 1024];
// static size_t feature_ix = 0;
char *buffer;

snd_pcm_t *initAlsa(snd_pcm_format_t);
void recognise(int);

int get_feature_data(size_t offset, size_t length, float *out_ptr)
{
    printf("%d, offset:%lu", buffer[0], offset);

    memcpy(out_ptr, buffer + offset, length * sizeof(char));
    return 0;
}

EI_IMPULSE_ERROR run_classifier(signal_t *, ei_impulse_result_t *, bool);

int main(int argc, char *argv[])
{

    int buffer_frames = 128;
    snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;
    snd_pcm_t *capture_handle;
    int err;
    capture_handle = initAlsa(format);

    int size = 128 * snd_pcm_format_width(format) / 8 * 2;
    buffer = (char *)malloc(size);

    fprintf(stdout, "buffer allocated\n");

    for (int i = 0; i < 1; ++i)
    {
        if ((err = snd_pcm_readi(capture_handle, buffer, buffer_frames)) != buffer_frames)
        {
            fprintf(stderr, "read from audio interface failed (%s)\n",
                    snd_strerror(err));
            exit(1);
        }

        recognise(size);

        // printf("%d \n", buffer[0]);
    }

    free(buffer);

    fprintf(stdout, "buffer freed\n");

    snd_pcm_close(capture_handle);
    fprintf(stdout, "audio interface closed\n");
}

void recognise(int size)
{
    signal_t signal;
    signal.total_length = size;
    signal.get_data = &get_feature_data;

    ei_impulse_result_t result;

    EI_IMPULSE_ERROR res = run_classifier(&signal, &result, true);
    printf("run_classifier returned: %d\n", res);

    printf("Begin output\n");

    // print the predictions
    printf("[");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
    {
        printf("%.5f", result.classification[ix].value);
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        printf(", ");
#else
        if (ix != EI_CLASSIFIER_LABEL_COUNT - 1)
        {
            printf(", ");
        }
#endif
    }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    printf("%.3f", result.anomaly);
#endif
    printf("]\n");

    printf("End output\n");
}

snd_pcm_t *initAlsa(snd_pcm_format_t format)
{
    int channels = 1;
    char *card = "hw:1";
    int err;
    snd_pcm_t *capture_handle;
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
