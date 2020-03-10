#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "numpy_types.h"
#include "ei_classifier_porting.h"
#include "ei_classifier_types.h"

static float features[1 * 1024 * 1024];
static size_t feature_ix = 0;

int get_feature_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

EI_IMPULSE_ERROR run_classifier(signal_t *, ei_impulse_result_t *, bool);
int signal_from_buffer(float *data, size_t data_size, signal_t *signal);

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Requires one parameter (a comma-separated list of raw features)\n");
        return 1;
    }

    char *token = strtok(argv[1], ",");
    while (token != NULL) {
        features[feature_ix++] = strtof(token, NULL);
        token = strtok(NULL, " ");
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
