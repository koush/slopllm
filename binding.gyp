{
  "targets": [
    {
      "target_name": "glm",
      "sources": [ "csrc/glm_ops.cpp" ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "csrc",
        "/usr/local/cuda/include"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "libraries": [
        "-L<(PRODUCT_DIR)",
        "-lglm_ops",
        "-L/usr/local/cuda/lib64",
        "-lcudart",
        "-lcublas"
      ],
      "ldflags": [ "-Wl,-rpath,\\$$ORIGIN" ],
      "actions": [
        {
          "action_name": "build_libglm",
          "inputs": [ "csrc/glm_ops.cu", "csrc/glm_flash.cu", "csrc/glm_fp8.cu", "csrc/glm_gdn.cu", "csrc/glm_sampling.cu" ],
          "outputs": [ "<(PRODUCT_DIR)/libglm_ops.so" ],
          "action": [
            "nvcc", "-O2", "-Xcompiler", "-fPIC", "-shared",
            "-gencode", "arch=compute_120a,code=sm_120a",
            "--expt-relaxed-constexpr",
            "-o", "<@(_outputs)", "<@(_inputs)",
            "-Ivendor/flashinfer/include",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64", "-lcublas", "-lcudart",
            "-Xcompiler", "-fPIC"
          ],
          "message": "Building libglm_ops.so"
        }
      ],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES"
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1
        }
      }
    }
  ]
}
