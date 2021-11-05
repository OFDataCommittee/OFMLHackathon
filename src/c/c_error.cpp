/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Hewlett Packard Enterprise
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
#include "srexception.h"
#include "srassert.h"

using namespace SmartRedis;

// The last error encountered
static smart_error __last_error = smart_error("no error");

// Store the last error encountered
extern "C"
void sr_set_last_error(const smart_error& last_error)
{
  // Check environment for debug level if we haven't done so yet
  static bool __debug_level_verbose = false;
  static bool __debug_level_checked = false;
  if (!__debug_level_checked)
  {
    __debug_level_checked = true;
     std::string dbgLevel(getenv("SMARTREDIS_DEBUG_LEVEL"));
     __debug_level_verbose = dbgLevel.compare("VERBOSE") == 0;
  }

  // Print out the error message if verbose
  if (__debug_level_verbose && sr_ok != last_error.to_error_code()) {
    printf("%s\n", last_error.what());
  }

  // Store the last error
  __last_error = last_error;
}

// Return the last error encountered
extern "C"
const char* sr_get_last_error()  {
  return __last_error.what();
}
