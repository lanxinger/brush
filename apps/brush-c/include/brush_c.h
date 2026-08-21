#ifndef BRUSH_C_H
#define BRUSH_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum BrushTrainExitCode {
    BrushTrainExitCodeSuccess = 0,
    BrushTrainExitCodeError = 1,
} BrushTrainExitCode;

typedef enum BrushProgressMessageTag {
    BrushProgressMessageNewProcess = 0,
    BrushProgressMessageTraining = 1,
    BrushProgressMessageDoneTraining = 2,
} BrushProgressMessageTag;

typedef struct BrushProgressMessageTrainingBody {
    uint32_t iter;
} BrushProgressMessageTrainingBody;

typedef struct BrushProgressMessage {
    BrushProgressMessageTag tag;
    union {
        BrushProgressMessageTrainingBody training;
    } payload;
} BrushProgressMessage;

typedef struct BrushTrainOptions {
    uint32_t total_train_steps;
    uint32_t refine_every;
    uint32_t max_resolution;
    uint32_t export_every;
    const char *output_path;
} BrushTrainOptions;

typedef struct BrushTrainOptionsV2 {
    uint32_t total_train_steps;
    uint32_t refine_every;
    uint32_t max_resolution;
    uint32_t export_every;
    uint32_t alpha_mode;
    uint32_t max_splats;
    const char *output_path;
} BrushTrainOptionsV2;

typedef void (*BrushProgressCallback)(BrushProgressMessage message, void *user_data);

/* Blocks the calling thread until training completes. */
BrushTrainExitCode train_and_save(
    const char *dataset_path,
    const BrushTrainOptions *options,
    BrushProgressCallback progress_callback,
    void *user_data
);

/*
 * Adds alpha_mode (0 auto, 1 masked, 2 transparent) and max_splats
 * (0 keeps Brush's default) without changing the original options ABI.
 */
BrushTrainExitCode train_and_save_v2(
    const char *dataset_path,
    const BrushTrainOptionsV2 *options,
    BrushProgressCallback progress_callback,
    void *user_data
);

#ifdef __cplusplus
}
#endif

#endif /* BRUSH_C_H */
