#include "ppl/common/log.h"

void CvtFp32ToFp16(int counts, void const *src, void *dst) {
    LOG(DEBUG) << "fp32 to fp16";
    auto src_ptr = (float*)src;
    auto dst_ptr = (__fp16*)dst;
    for (int i = 0; i < counts; i += 1) {
        dst_ptr[i] = src_ptr[i];
    }
}

void CvtFp16ToFp32(int counts, void const *src, void *dst) {
    LOG(DEBUG) << "fp16 to fp32";
    auto src_ptr = (__fp16*)src;
    auto dst_ptr = (float*)dst;
    for (int i = 0; i < counts; i += 1) {
        dst_ptr[i] = src_ptr[i];
    }
}

