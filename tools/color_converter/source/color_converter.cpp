#include "color_converter.h"

static constexpr const uint32_t stb_fp32_to_srgb8_tab4[104] =
{
    0x0073000d, 0x007a000d, 0x0080000d, 0x0087000d, 0x008d000d, 0x0094000d, 0x009a000d, 0x00a1000d,
    0x00a7001a, 0x00b4001a, 0x00c1001a, 0x00ce001a, 0x00da001a, 0x00e7001a, 0x00f4001a, 0x0101001a,
    0x010e0033, 0x01280033, 0x01410033, 0x015b0033, 0x01750033, 0x018f0033, 0x01a80033, 0x01c20033,
    0x01dc0067, 0x020f0067, 0x02430067, 0x02760067, 0x02aa0067, 0x02dd0067, 0x03110067, 0x03440067,
    0x037800ce, 0x03df00ce, 0x044600ce, 0x04ad00ce, 0x051400ce, 0x057b00c5, 0x05dd00bc, 0x063b00b5,
    0x06970158, 0x07420142, 0x07e30130, 0x087b0120, 0x090b0112, 0x09940106, 0x0a1700fc, 0x0a9500f2,
    0x0b0f01cb, 0x0bf401ae, 0x0ccb0195, 0x0d950180, 0x0e56016e, 0x0f0d015e, 0x0fbc0150, 0x10630143,
    0x11070264, 0x1238023e, 0x1357021d, 0x14660201, 0x156601e9, 0x165a01d3, 0x174401c0, 0x182401af,
    0x18fe0331, 0x1a9602fe, 0x1c1502d2, 0x1d7e02ad, 0x1ed4028d, 0x201a0270, 0x21520256, 0x227d0240,
    0x239f0443, 0x25c003fe, 0x27bf03c4, 0x29a10392, 0x2b6a0367, 0x2d1d0341, 0x2ebe031f, 0x304d0300,
    0x31d105b0, 0x34a80555, 0x37520507, 0x39d504c5, 0x3c37048b, 0x3e7c0458, 0x40a8042a, 0x42bd0401,
    0x44c20798, 0x488e071e, 0x4c1c06b6, 0x4f76065d, 0x52a50610, 0x55ac05cc, 0x5892058f, 0x5b590559,
    0x5e0c0a23, 0x631c0980, 0x67db08f6, 0x6c55087f, 0x70940818, 0x74a007bd, 0x787d076c, 0x7c330723,
};

uint32_t convert_linear_to_srgb_sse2(const float* in_linear)
{
    const __m128 in_bgra = _mm_loadu_ps(in_linear);

    // Clamp to [2^(-13), 1-eps]; these two values map to 0 and 1, respectively.
    // This clamping logic is carefully written so that NaNs map to 0.
    //
    // We do this clamping on all four color channels, even though we later handle A differently; 
    // this does not change the results for A: 2^(-13) rounds to 0 in U8, 
    // and 1-eps rounds to 255 in U8, so these are OK endpoints to use.
    const __m128 almost_one = _mm_castsi128_ps(_mm_set1_epi32(0x3f7fffff)); // 1-eps
    const __m128i min_int_value = _mm_set1_epi32((127 - 13) << 23);
    const __m128 min_float_value = _mm_castsi128_ps(min_int_value);

    const __m128 in_clamped = _mm_min_ps(_mm_max_ps(in_bgra, min_float_value), almost_one);

    // Set up for the table lookup
    // This computes a 3-vector of table indices. The above clamping
    // ensures that the values in question are in [0,13*8-1]=[0,103].
    const __m128i tab_index = _mm_srli_epi32(_mm_sub_epi32(_mm_castps_si128(in_clamped), min_int_value), 20);

    // Do the 4 table lookups with regular loads. We can use PEXTRW (SSE2)
    // to grab the 3 indices from lanes 1-3, lane 0 we can just get via MOVD.
    // The latter gives us a full 32 bits, not 16 like the other ones, but given our value range either works.
    const __m128i table_value_b = _mm_cvtsi32_si128(stb_fp32_to_srgb8_tab4[(uint32_t)_mm_cvtsi128_si32(tab_index)]);
    const __m128i table_value_g = _mm_cvtsi32_si128(stb_fp32_to_srgb8_tab4[(uint32_t)_mm_extract_epi16(tab_index, 2)]);
    const __m128i table_value_r = _mm_cvtsi32_si128(stb_fp32_to_srgb8_tab4[(uint32_t)_mm_extract_epi16(tab_index, 4)]);

    // Merge the four values we just loaded back into a 3-vector (gather complete!)
    const __m128i table_value_bg = _mm_unpacklo_epi32(table_value_b, table_value_g);
    const __m128i table_values_bgr = _mm_unpacklo_epi64(table_value_bg, table_value_r); // This leaves A=0, which suits us

    // Grab the mantissa bits into the low 16 bits of each 32b lane, and set up 512 in the high 16 bits of each 32b lane, 
    // which is how the bias values in the table are meant to be scaled.
    //
    // We grab mantissa bits [12,19] for the lerp.
    const __m128i mantissa_lerp_factor = _mm_and_si128(_mm_srli_epi32(_mm_castps_si128(in_clamped), 12), _mm_set1_epi32(0xff));
    const __m128i final_multiplier = _mm_or_si128(mantissa_lerp_factor, _mm_set1_epi32(512 << 16));

    // In the table:
    //    (bias>>9) was stored in the high 16 bits
    //    scale was stored in the low 16 bits
    //    t = (mantissa >> 12) & 0xff
    //
    // then we want ((bias + scale*t) >> 16).
    // Except for the final shift, that's a single PMADDWD:
    // const __m128i interpolated_rgb = _mm_srli_epi32(_mm_madd_epi16(table_values_rgb, final_multiplier), 16);
    const __m128i interpolated_bgr = _mm_srli_epi32(_mm_madd_epi16(table_values_bgr, final_multiplier), 16);

    // Finally, A gets done directly, via (int)(A * 255.f + 0.5f)
    // We zero out the non-A channels by multiplying by 0; our clamping earlier took care of NaNs/infinites, so this is fine.
    const __m128 scaled_biased_alpha = _mm_add_ps(_mm_mul_ps(in_clamped, _mm_setr_ps(0.f, 0.f, 0.f, 255.f)), _mm_set1_ps(0.5f));
    const __m128i final_alpha = _mm_cvttps_epi32(scaled_biased_alpha);

    // Merge A into the result, reorder to BGRA, then pack down to bytes and store!
    // interpolated_rgb has lane 3=0, and ComputedA has the first three lanes zero, so we can just OR them together.
    const __m128i final_bgra = _mm_or_si128(interpolated_bgr, final_alpha);

    const __m128i packed16 = _mm_packs_epi32(final_bgra, final_bgra);
    const __m128i packed8 = _mm_packus_epi16(packed16, packed16);

    return (uint32_t)_mm_cvtsi128_si32(packed8);
}

void convert_linear_to_srgb(uint8_t* srgb_array, const float* linear_array, int32_t pixel_count) // bgra
{
    if (pixel_count % 4 == 0)
    {
        for (int32_t i = 0; i < pixel_count; i += 4)
        {
            uint32_t bits = convert_linear_to_srgb_sse2(linear_array + i);
            srgb_array[i]     = (uint8_t)((bits >> 0)  & 0x000000FF);
            srgb_array[i + 1] = (uint8_t)((bits >> 8)  & 0x000000FF);
            srgb_array[i + 2] = (uint8_t)((bits >> 16) & 0x000000FF);
            srgb_array[i + 3] = (uint8_t)((bits >> 24) & 0x000000FF);
        }
    }
}

// void ConvertLinearToSRGB_RGBA(uint8_t* srgb_array, const float* linear_array, int32_t pixel_count)
// {
//     if (pixel_count % 4 == 0)
//     {
//         for (int32_t i = 0; i < pixel_count; i += 4)
//         {
//             uint32_t bits = convert_linear_to_srgb_sse2(linear_array + i);
//             srgb_array[i]     = (uint8_t)((bits >> 16) & 0x000000FF);
//             srgb_array[i + 1] = (uint8_t)((bits >> 8)  & 0x000000FF);
//             srgb_array[i + 2] = (uint8_t)((bits >> 0)  & 0x000000FF);
//             srgb_array[i + 3] = (uint8_t)((bits >> 24) & 0x000000FF);
//         }
//     }
// }
