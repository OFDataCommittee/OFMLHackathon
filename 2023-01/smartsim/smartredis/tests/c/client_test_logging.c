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

#include "c_client.h"
#include "c_dataset.h"
#include "c_logcontext.h"
#include "c_client_test_utils.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "stdint.h"
#include "srexception.h"
#include "c_logger.h"

bool cluster = true;

#define TEST_LOG(logtype, context, loglevel, logmessage) \
log_##logtype(context, loglevel, logmessage, strlen(logmessage))

#define TEST_LOG_STRING(logtype, context, loglevel, logmessage) \
log_##logtype##_string(\
  context, strlen(context), loglevel, logmessage, strlen(logmessage))

int main(int argc, char* argv[])
{
  int result = 0;
  void* client = NULL;
  void* dataset = NULL;
  void* logcontext = NULL;
  const char* ctx_client = "client_test_logging (client)";
  const char* ctx_dataset = "client_test_logging (dataset)";
  const char* ctx_logcontext = "client_test_logging (logcontext)";
  const char* ctx_string = "client_test_logging (string)";
  size_t ctx_client_len = strlen(ctx_client);
  size_t ctx_dataset_len = strlen(ctx_dataset);
  size_t ctx_logcontext_len = strlen(ctx_logcontext);

  // Initialize client, dataset, logcontext
  if (SRNoError != SmartRedisCClient(
    use_cluster(), ctx_client, ctx_client_len,
    &client) || NULL == client) {
    return -1;
  }
  if (SRNoError != CDataSet(
    ctx_dataset, ctx_dataset_len, &dataset) || NULL == dataset) {
    return -1;
  }
  if (SRNoError != SmartRedisCLogContext(
    ctx_logcontext, ctx_logcontext_len, &logcontext) || NULL == logcontext) {
    return -1;
  }

  // Log stuff against a client
  TEST_LOG(data, client, LLQuiet,
    "This is data logged against the client at the Quiet level");
  TEST_LOG(data, client, LLInfo,
    "This is data logged against the client at the Info level");
  TEST_LOG(data, client, LLDebug,
    "This is data logged against the client at the Debug level");
  TEST_LOG(data, client, LLDeveloper,
    "This is data logged against the client at the Developer level");

  TEST_LOG(warning, client, LLQuiet,
    "This is a warning logged against the client at the Quiet level");
  TEST_LOG(warning, client, LLInfo,
    "This is a warning logged against the client at the Info level");
  TEST_LOG(warning, client, LLDebug,
    "This is a warning logged against the client at the Debug level");
  TEST_LOG(warning, client, LLDeveloper,
    "This is a warning logged against the client at the Developer level");

  TEST_LOG(error, client, LLQuiet,
    "This is an error logged against the client at the Quiet level");
  TEST_LOG(error, client, LLInfo,
    "This is an error logged against the client at the Info level");
  TEST_LOG(error, client, LLDebug,
    "This is an error logged against the client at the Debug level");
  TEST_LOG(error, client, LLDeveloper,
    "This is an error logged against the client at the Developer level");

  // Log stuff against a dataset
  TEST_LOG(data, dataset, LLQuiet,
    "This is data logged against the dataset at the Quiet level");
  TEST_LOG(data, dataset, LLInfo,
    "This is data logged against the dataset at the Info level");
  TEST_LOG(data, dataset, LLDebug,
    "This is data logged against the dataset at the Debug level");
  TEST_LOG(data, dataset, LLDeveloper,
    "This is data logged against the dataset at the Developer level");

  TEST_LOG(warning, dataset, LLQuiet,
    "This is a warning logged against the dataset at the Quiet level");
  TEST_LOG(warning, dataset, LLInfo,
    "This is a warning logged against the dataset at the Info level");
  TEST_LOG(warning, dataset, LLDebug,
    "This is a warning logged against the dataset at the Debug level");
  TEST_LOG(warning, dataset, LLDeveloper,
    "This is a warning logged against the dataset at the Developer level");

  TEST_LOG(error, dataset, LLQuiet,
    "This is an error logged against the dataset at the Quiet level");
  TEST_LOG(error, dataset, LLInfo,
    "This is an error logged against the dataset at the Info level");
  TEST_LOG(error, dataset, LLDebug,
    "This is an error logged against the dataset at the Debug level");
  TEST_LOG(error, dataset, LLDeveloper,
    "This is an error logged against the dataset at the Developer level");

  // Log stuff against a logcontext
  TEST_LOG(data, logcontext, LLQuiet,
    "This is data logged against the logcontext at the Quiet level");
  TEST_LOG(data, logcontext, LLInfo,
    "This is data logged against the logcontext at the Info level");
  TEST_LOG(data, logcontext, LLDebug,
    "This is data logged against the logcontext at the Debug level");
  TEST_LOG(data, logcontext, LLDeveloper,
    "This is data logged against the logcontext at the Developer level");

  TEST_LOG(warning, logcontext, LLQuiet,
    "This is a warning logged against the logcontext at the Quiet level");
  TEST_LOG(warning, logcontext, LLInfo,
    "This is a warning logged against the logcontext at the Info level");
  TEST_LOG(warning, logcontext, LLDebug,
    "This is a warning logged against the logcontext at the Debug level");
  TEST_LOG(warning, logcontext, LLDeveloper,
    "This is a warning logged against the logcontext at the Developer level");

  TEST_LOG(error, logcontext, LLQuiet,
    "This is an error logged against the logcontext at the Quiet level");
  TEST_LOG(error, logcontext, LLInfo,
    "This is an error logged against the logcontext at the Info level");
  TEST_LOG(error, logcontext, LLDebug,
    "This is an error logged against the logcontext at the Debug level");
  TEST_LOG(error, logcontext, LLDeveloper,
    "This is an error logged against the logcontext at the Developer level");

   // Log stuff against a string
  TEST_LOG_STRING(data, ctx_string, LLQuiet,
    "This is data logged against a string at the Quiet level");
  TEST_LOG_STRING(data, ctx_string, LLInfo,
    "This is data logged against a string at the Info level");
  TEST_LOG_STRING(data, ctx_string, LLDebug,
    "This is data logged against a string at the Debug level");
  TEST_LOG_STRING(data, ctx_string, LLDeveloper,
    "This is data logged against a string at the Developer level");

  TEST_LOG_STRING(warning, ctx_string, LLQuiet,
    "This is a warning logged against a string at the Quiet level");
  TEST_LOG_STRING(warning, ctx_string, LLInfo,
    "This is a warning logged against a string at the Info level");
  TEST_LOG_STRING(warning, ctx_string, LLDebug,
    "This is a warning logged against a string at the Debug level");
  TEST_LOG_STRING(warning, ctx_string, LLDeveloper,
    "This is a warning logged against a string at the Developer level");

  TEST_LOG_STRING(error, ctx_string, LLQuiet,
    "This is an error logged against a string at the Quiet level");
  TEST_LOG_STRING(error, ctx_string, LLInfo,
    "This is an error logged against a string at the Info level");
  TEST_LOG_STRING(error, ctx_string, LLDebug,
    "This is an error logged against a string at the Debug level");
  TEST_LOG_STRING(error, ctx_string, LLDeveloper,
    "This is an error logged against a string at the Developer level");


  // Done
  printf("Test passed: %s\n", result == 0 ? "YES" : "NO");
  return result;
}
