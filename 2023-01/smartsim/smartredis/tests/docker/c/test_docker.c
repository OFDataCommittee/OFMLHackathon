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

#include "stdio.h"
#include "string.h"

#include "c_client.h"
#include "srexception.h"
#include "sr_enums.h"

int main(int argc, char* argv[]) {

    void* client = NULL;
    const char* logger_name = "test_docker";
    size_t cid_len = strlen(logger_name);

    SRError return_code = SRNoError;
    return_code = SmartRedisCClient(false, logger_name, cid_len, &client);

    if (return_code != SRNoError) {
        return -1;
    }

    char* key = "c_docker_tensor";
    size_t key_length = strlen(key);

    double tensor[3] = {1.0, 2.0, 3.0};
    SRTensorType type = SRTensorTypeDouble;
    SRMemoryLayout layout = SRMemLayoutContiguous;
    size_t dims[1] = {3};
    size_t n_dims = 1;

    return_code = put_tensor(client, key, key_length,
                            (void*)(&tensor), (const size_t*)(&dims),
                            n_dims, type, layout);

    if (return_code != SRNoError) {
        return -1;
    }

    double returned[3];

    return_code = unpack_tensor(client, key, key_length,
                                (void*)(&returned), (const size_t*)(&dims),
                                n_dims, type, layout);

    if (return_code != SRNoError) {
        return -1;
    }

    for (size_t i = 0; i < 3; i++) {
        if (returned[i] != tensor[i])
            return -1;
    }

    return 0;
}