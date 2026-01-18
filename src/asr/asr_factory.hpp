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

#include "asr/asr_interface.hpp"
#include <memory>

typedef std::shared_ptr<ASRInterface>   ASRIfPtr;

enum class ASR_TYPE_E {
    WHISPER = 0
};

class ASRFactory {
public:
    static ASRIfPtr create(ASR_TYPE_E type) {
        return nullptr;
    }
};