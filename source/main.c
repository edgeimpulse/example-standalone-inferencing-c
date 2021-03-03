#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "edge-impulse-sdk/dsp/numpy_types.h"
#include "edge-impulse-sdk/porting/ei_classifier_porting.h"
#include "edge-impulse-sdk/classifier/ei_classifier_types.h"

static float features[1 * 1024 * 1024];
static size_t feature_ix = 0;

int get_feature_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

int read_features_file(const char *filename) {
    char buffer[1 * 1024 * 1024] = { 0 };
    FILE *f = (FILE*)fopen(filename, "r");
    if (!f) {
        printf("Cannot open file %s\n", filename);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);
    fread(buffer, 1, size, f);
    fclose(f);

    char *token = strtok(buffer, ",");
    while (token != NULL) {
        features[feature_ix++] = strtof(token, NULL);
        token = strtok(NULL, " ");
    }
    return 0;
}

EI_IMPULSE_ERROR run_classifier(signal_t *, ei_impulse_result_t *, bool);

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Requires one parameter (a comma-separated list of raw features, or a file pointing at raw features)\n");
        return 1;
    }

    if (!strchr(argv[1], ' ') && strchr(argv[1], '.')) { // looks like a filename
        int r = read_features_file(argv[1]);
        if (r != 0) {
            return 1;
        }
    }
    else { // looks like a features array passed in as an argument
        char *token = strtok(argv[1], ",");
        while (token != NULL) {
            features[feature_ix++] = strtof(token, NULL);
            token = strtok(NULL, ",");
        }
    }

    if (feature_ix != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        printf("The size of your 'features' array is not correct. Expected %d items, but had %lu\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, feature_ix);
        return 1;
    }

    signal_t signal;
    signal.total_length = feature_ix;
    signal.get_data = &get_feature_data;

    ei_impulse_result_t result;

    EI_IMPULSE_ERROR res = run_classifier(&signal, &result, true);
    printf("run_classifier returned: %d\n", res);

    printf("Begin output\n");

    // print the predictions
    printf("[");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        printf("%.5f", result.classification[ix].value);
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        printf(", ");
#else
        if (ix != EI_CLASSIFIER_LABEL_COUNT - 1) {
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
