
/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SMARTREDIS_C_LOGCONTEXT_H
#define SMARTREDIS_C_LOGCONTEXT_H

#include "logcontext.h"
#include "sr_enums.h"
#include "srexception.h"

///@file
///\brief C-wrappers for the C++ LogContext class

#ifdef __cplusplus
extern "C" {
#endif

/*!
*   \brief C-LogContext constructor
*   \param context Logging context (string to prefix log entries with)
*   \param context_length The length of the context name string,
*                         excluding null terminating character
*   \param new_logcontext Receives the new logcontext
*   \return Returns SRNoError on success or an error code on failure
*/
SRError SmartRedisCLogContext(
    const char* context,
    const size_t context_length,
    void** new_logcontext);

/*!
*   \brief C-LogContext destructor
*   \param logcontext A pointer to the logcontext to release. The logcontext
*                     is set to NULL on completion
*   \return Returns SRNoError on success or an error code on failure
*/
SRError DeallocateLogContext(void** logcontext);

#ifdef __cplusplus
}
#endif
#endif // SMARTREDIS_C_LOGCONTEXT_H
