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
        "-lcublas",
        "-lnccl"
      ],
      "ldflags": [ "-Wl,-rpath,\\$$ORIGIN" ],
      "actions": [
        {
          "action_name": "build_libglm",
          "inputs": [ "Makefile", "csrc/glm_ops.cu", "csrc/glm_flash.cu", "csrc/glm_gemv.cu", "csrc/glm_gdn.cu", "csrc/glm_sampling.cu", "csrc/glm_p2p.cu", "csrc/glm_context_parallel.cu", "csrc/glm_grouped_moe.cu", "csrc/glm_mma_moe.cu", "csrc/glm_mma_moe_pc.cu", "csrc/glm_mma_moe_coop.cu", "csrc/glm_indexer.cu", "csrc/glm_nccl.cpp", "csrc/glm_device.cpp" ],
          "outputs": [ "<(PRODUCT_DIR)/libglm_ops.so" ],
          "action": [
            "make", "-j", "<!@(nproc)"
          ],
          "message": "Building libglm_ops.so (parallel)"
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
