void(ACFunction)();
int microphone_audio_signal_get_data(size_t, size_t, float *);
void microphone_inference_end();
int run_classifier(signal_t *, ei_impulse_result_t *, bool);