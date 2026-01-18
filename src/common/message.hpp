/**************************************************************************************************
 *
 * Copyright (c) 2019-2025 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/
#pragma once

#include <stdint.h>

typedef struct _AudioMessage {
    uint8_t channels;
    uint8_t bits_per_sample;
    uint16_t sample_rate;
    uint32_t num_samples;
    uint8_t data[];
} AudioMessage;

typedef struct _TextMessage {
    uint32_t text_length;     // 文本长度（字节数）
    uint16_t encoding;        // 0=UTF8, 1=GBK, 2=UTF16
    uint8_t  is_final;        // 是否最终结果
    uint32_t language;        // 语言代码 0x7A68="zh", 0x656E="en"
    char data[];
} TextMessage;