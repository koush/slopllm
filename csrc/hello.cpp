#include <napi.h>
#include <stdio.h>
#include "hello_cuda.h"

static Napi::Value Hello(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  printf("Hello from native addon!\n");
  launch_hello_kernel();
  return Napi::String::New(env, "hello from native addon");
}

static Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "hello"),
              Napi::Function::New(env, Hello));
  return exports;
}

NODE_API_MODULE(hello_cuda, Init)
