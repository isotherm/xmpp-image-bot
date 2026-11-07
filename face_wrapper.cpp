#include "facedetectcnn.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>

// libfacedetection's public result buffer stores a count (int), followed
// by records of 142 int16 values: confidence, x, y, width, height, ...
extern "C" int detect_faces_bgr(const unsigned char* bgr, int width, int height,
                                 int* out, int max_faces) {
    if (!bgr || !out || width <= 0 || height <= 0 || max_faces <= 0 ||
        width > 8192 || height > 8192) return -1;
    alignas(16) unsigned char results[FACEDETECTION_RESULT_BUFFER_SIZE] = {};
    int* found = facedetect_cnn(results, const_cast<unsigned char*>(bgr),
                                width, height, width * 3);
    //fprintf(stderr, "libfacedetection returned %d faces\n", found[0]);
    if (!found) return -2;
    int count = std::min(std::max(found[0], 0), std::min(max_faces, FACEDETECTION_RESULT_MAX_FACES));
    for (int i = 0; i < count; ++i) {
        const auto* record = reinterpret_cast<const int16_t*>(found + 1) + i * FACEDETECTION_RESULT_STRIDE_SHORTS;
        for (int j = 0; j < 5; ++j) out[i * 5 + j] = record[j];
    }
    return count;
}
