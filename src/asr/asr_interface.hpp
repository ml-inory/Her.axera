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

#include "common/message.hpp"
#include <string>

class ASRInterface {
public:
    virtual bool init(const char* model_path) = 0;
    virtual void uninit(void) = 0;
    virtual bool run(const AudioMessage& audio_msg, TextMessage& text_msg) = 0;
};