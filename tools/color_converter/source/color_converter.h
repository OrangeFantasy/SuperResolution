#pragma once

#if defined(_MSC_VER)
    #define DLLEXPORT __declspec(dllexport)
    #define DLLIMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    #define DLLEXPORT __attribute__((visibility("default")))
    #define DLLIMPORT
#endif

#include <cstdint>
#include <emmintrin.h>

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    DLLEXPORT void convert_linear_to_srgb(uint8_t* OutSRGBColor, const float* InLiearColor, int32_t Size);

    // DLLEXPORT void ConvertLinearToSRGB_RGBA(uint8_t* OutSRGBColor, const float* InLiearColor, int32_t Size);

#ifdef __cplusplus
}
#endif // __cplusplus
